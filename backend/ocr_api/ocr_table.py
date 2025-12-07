import os
import argparse
import json
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageDraw
from paddleocr import PaddleOCR, TableRecognitionPipelineV2
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


# === Lokalne utilsy (zamiast importu z main.py / check_table_detection.py) ===

MAX_SIDE_LIMIT = 4000
OVERLAP = 100
# Parametry heurystyki wykrywania nagłówka
HEADER_MIN_MATCH_IN_LINE = 2  # minimalna liczba dopasowanych nagłówków w jednej linii, aby uznać ją za wiersz nagłówkowy
HEADER_TOP_MARGIN = 8         # ile pikseli nad wykrytym wierszem nagłówka włączyć do ROI
HEADER_ROW_MERGE_GAP = 16     # maks. pionowy odstęp (px) przy łączeniu sąsiadujących wierszy nagłówkowych

# Parametry łączenia bliskich słów (by poprawić wykrywanie nagłówków)
WORD_MERGE_GAP_MIN_PX = 8            # minimalny odstęp (px), poniżej którego łączymy słowa
WORD_MERGE_GAP_REL_TO_HEIGHT = 0.45  # część mediany wysokości linii, która wyznacza próg łączenia

# Parametry do łączenia par nagłówek → wartość (fallback poza-tabelowy)
HEADER_SIM_THRESHOLD = 0.74          # próg podobieństwa dla zmapowania nagłówka do HEADER_MAP
HEADER_LIKE_SIM_THRESHOLD = 0.62     # niższy próg traktowania frazy jako „nagłówkopodobnej”
PAIR_SAME_ROW_GAP_PX = 160           # maksymalny poziomy odstęp do wartości w tej samej linii
PAIR_BELOW_MAX_ROWS = 2              # szukaj wartości maksymalnie w tylu kolejnych wierszach poniżej
PAIR_X_OVERLAP_MIN_RATIO = 0.15      # minimalny wsp. pokrycia w poziomie przy szukaniu wiersza poniżej

# Mapa nagłówków -> ścieżki pól w docelowym schemacie elementu
HEADER_MAP: Dict[str, str] = {
    "Rzecz": "item",
    "Rzecz znaleziona": "item",
    "znaleziona Rzecz": "item",
    "Nazwa": "item",
    "Opis rzeczy": "item",
    "Nazwa - opis rzeczy": "item",
    "WYKAZ RZECZY ZNALEZIONYCH": "item",

    "Data znalezienia": "foundDateTime",
    "Data znalezienia rzeczy": "foundDateTime",

    "Miejsce znalezienia": "location",
    "Miejsce znalezienia - skąd przyjęto": "location",

    "Informację wytworzył(-a)": "metadata.createdBy",
    "Data wytworzenia": "metadata.createdAt",
    "Data": "metadata.createdAt",
    "Ogłoszenie wprowadził(-a)": "metadata.enteredBy",
    "Data wprowadzenia": "metadata.enteredAt",
    "Na stronie biuletynu od": "metadata.publishedFrom",
    "Na stronie biuletynu do": "metadata.publishedTo",
    "Na podstawie": "metadata.Description",
    "Nr sprawy": "metadata.Description",
    "Uwagi": "metadata.Description",
    "L.p.": "metadata.rowNumber",
    "Lp.": "metadata.rowNumber",
    "BIP": "metadata.bipNumber",
    "Data przyjęcia do biura": "metadata.receivedAt",
    "Data wpływu rzeczy do Biura Rzeczy Znalezionych": "metadata.receivedAt",
    "DATA KOMUNIKATU": "metadata.createdAt",
    "Rodzaj dokumentu / kategoria": "metadata.category",
    "Link do szczegółów / załącznika": "metadata.detailsUrl",
    "Status": "metadata.status",
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_box_points(box) -> List[List[float]]:
    """Konwertuje bbox (8 liczb lub lista punktów) do listy 4 punktów [x,y]."""
    if isinstance(box, (list, tuple)) and len(box) == 8 and isinstance(box[0], (int, float)):
        return [[float(box[i]), float(box[i + 1])] for i in range(0, 8, 2)]
    return [[float(x), float(y)] for x, y in box]


def polygon_to_rect(box: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def rect_to_box(rect: Tuple[float, float, float, float]) -> List[List[float]]:
    x1, y1, x2, y2 = rect
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def draw_boxes_on_image(im: Image.Image, boxes, color=(255, 0, 0), width: int = 2) -> Image.Image:
    vis = im.copy()
    draw = ImageDraw.Draw(vis)
    for box in boxes:
        # Obsłuż format spłaszczony [x1,y1,...,x4,y4] jak i listę par [[x,y],...]
        if isinstance(box, (list, tuple)) and box and isinstance(box[0], (int, float)) and len(box) == 8:
            pts = [(float(box[i]), float(box[i + 1])) for i in range(0, 8, 2)]
        else:
            pts = [(float(x), float(y)) for x, y in box]
        pts.append(pts[0])
        draw.line(pts, fill=color, width=width)
    return vis


def run_ocr_on_tile(ocr: PaddleOCR, img_array: np.ndarray):
    outputs = ocr.predict(img_array)
    if not outputs:
        return [], [], []
    res = outputs[0]
    boxes = res.get("rec_polys", []) or []
    texts = res.get("rec_texts", []) or []
    scores = res.get("rec_scores", []) or []
    return boxes, texts, scores


def translate_box(box, dx: int, dy: int):
    return [[float(pt[0]) + dx, float(pt[1]) + dy] for pt in box]


def ocr_with_tiling_on_image(ocr: PaddleOCR, im: Image.Image, offset: Tuple[int, int] = (0, 0)):
    """
    Uruchamia OCR z podziałem na kafelki dla obrazu PIL (RGB).
    Zwraca globalne bboxy przesunięte o offset.
    """
    w, h = im.size

    all_boxes = []
    all_texts = []
    all_scores = []

    if max(w, h) <= MAX_SIDE_LIMIT:
        img_np = np.array(im)
        boxes, texts, scores = run_ocr_on_tile(ocr, img_np)
        for box, txt, score in zip(boxes, texts, scores):
            gbox = translate_box(box, offset[0], offset[1])
            all_boxes.append(gbox)
            all_texts.append(txt if not isinstance(txt, tuple) else txt[0])
            all_scores.append(float(score))
        return all_boxes, all_texts, all_scores, im

    stride_x = max(1, MAX_SIDE_LIMIT - OVERLAP)
    stride_y = max(1, MAX_SIDE_LIMIT - OVERLAP)

    y0 = 0
    while y0 < h:
        y1 = min(h, y0 + MAX_SIDE_LIMIT)
        if y1 == h and (y1 - y0) < MAX_SIDE_LIMIT and y0 > 0:
            y0 = max(0, h - MAX_SIDE_LIMIT)
            y1 = h
        x0 = 0
        while x0 < w:
            x1 = min(w, x0 + MAX_SIDE_LIMIT)
            if x1 == w and (x1 - x0) < MAX_SIDE_LIMIT and x0 > 0:
                x0 = max(0, w - MAX_SIDE_LIMIT)
                x1 = w

            tile = im.crop((x0, y0, x1, y1))
            boxes, texts, scores = run_ocr_on_tile(ocr, np.array(tile))
            for box, txt, score in zip(boxes, texts, scores):
                gbox = translate_box(box, x0 + offset[0], y0 + offset[1])
                all_boxes.append(gbox)
                all_texts.append(txt if not isinstance(txt, tuple) else txt[0])
                all_scores.append(float(score))

            if x1 == w:
                break
            x0 += stride_x

        if y1 == h:
            break
        y0 += stride_y

    return all_boxes, all_texts, all_scores, im


def capture_screenshot(url: str, output_path: str):
    """Pobiera pełnoekranowy screenshot strony pod wskazanym URL."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--start-maximized")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    driver.get(url)
    total_width = driver.execute_script("return document.body.offsetWidth")
    total_height = driver.execute_script("return document.body.parentNode.scrollHeight")
    driver.set_window_size(total_width, total_height)
    driver.save_screenshot(output_path)
    driver.quit()
    print(f"Screenshot saved to {output_path}")


def is_url(path_or_url: str) -> bool:
    p = path_or_url.strip().lower()
    return p.startswith("http://") or p.startswith("https://")


def union_rect(rects: List[Tuple[float, float, float, float]]) -> Tuple[int, int, int, int]:
    if not rects:
        raise ValueError("Brak prostokątów do złączenia")
    x1 = min(r[0] for r in rects)
    y1 = min(r[1] for r in rects)
    x2 = max(r[2] for r in rects)
    y2 = max(r[3] for r in rects)
    # Zwróć inty, zaokrąglając do krawędzi pikseli
    return int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))


def detect_table_polygons_v2(pipeline: TableRecognitionPipelineV2, img_path: str) -> List[List[List[float]]]:
    """
    Zwraca listę wielokątów (4 punkty) wyznaczających obszary tabel.
    Preferuje obszary z detekcji layoutu (layout_det_res -> label=="table"),
    a jeśli ich brak, próbuje użyć klucza 'bbox' ze struktury wynikowej.
    """
    polys: List[List[List[float]]] = []
    output = pipeline.predict(img_path)
    if not output:
        return polys
    for res in output:
        res_data = None
        if hasattr(res, "res"):
            res_data = res.res
        elif isinstance(res, dict):
            res_data = res.get("res", res)
        if not res_data:
            continue

        # 1) Spróbuj wyciągnąć obszary tabel z layout_det_res (jak w table_res.json)
        try:
            layout = res_data.get("layout_det_res") or {}
            boxes = layout.get("boxes") or []
            for item in boxes:
                label = item.get("label") or item.get("category")
                if str(label).lower() == "table":
                    coord = item.get("coordinate") or item.get("box")
                    if isinstance(coord, (list, tuple)) and len(coord) == 4:
                        x1, y1, x2, y2 = [float(c) for c in coord]
                        polys.append(rect_to_box((x1, y1, x2, y2)))
        except Exception:
            pass

        # 2) Jeśli jest dostępny klucz 'bbox' (polygony tabel), również je dołącz
        bboxes = res_data.get("bbox") or []
        for b in bboxes:
            try:
                polys.append(normalize_box_points(b))
            except Exception:
                continue
    return polys


def _box_rect(box: List[List[float]]):
    return polygon_to_rect(box)


def _median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _cluster_rows_by_y(boxes: List[List[List[float]]]):
    # Przyjmujemy, że boxes to lista poligonów 4-punktowych
    items = []
    for idx, b in enumerate(boxes):
        x1, y1, x2, y2 = _box_rect(b)
        h = max(1.0, y2 - y1)
        yc = (y1 + y2) / 2.0
        items.append({"idx": idx, "box": b, "rect": (x1, y1, x2, y2), "h": h, "yc": yc})
    if not items:
        return []
    items.sort(key=lambda it: it["yc"])  # sortuj po środku Y
    med_h = _median([it["h"] for it in items]) or 1.0
    thr = max(4.0, med_h * 0.6)
    rows = []
    current = [items[0]]
    current_y = items[0]["yc"]
    for it in items[1:]:
        if abs(it["yc"] - current_y) <= thr:
            current.append(it)
            # aktualizuj średnią y bieżącego wiersza
            current_y = sum(x["yc"] for x in current) / len(current)
        else:
            rows.append(current)
            current = [it]
            current_y = it["yc"]
    rows.append(current)
    # w każdym wierszu posortuj po x środku
    for r in rows:
        r.sort(key=lambda it: (it["rect"][0] + it["rect"][2]) / 2.0)
    return rows


def _merge_close_words_in_row(row_items, idx_to_text: Dict[int, str]):
    """
    Łączy kolejne elementy w wierszu, jeśli odstęp poziomy między nimi jest mały.
    Zwraca listę segmentów: [{"text": str, "rect": (x1,y1,x2,y2)}].
    """
    if not row_items:
        return []
    # posortuj po x1
    items = sorted(row_items, key=lambda it: it["rect"][0])
    # mediana wysokości wiersza
    heights = [max(1.0, it["rect"][3] - it["rect"][1]) for it in items]
    med_h = _median(heights) or 1.0
    thr = max(float(WORD_MERGE_GAP_MIN_PX), float(med_h) * float(WORD_MERGE_GAP_REL_TO_HEIGHT))

    segments = []
    # zainicjuj pierwszym elementem
    it0 = items[0]
    x1, y1, x2, y2 = it0["rect"]
    cur_text = _norm_header(idx_to_text.get(it0["idx"], ""))
    cur_rect = [float(x1), float(y1), float(x2), float(y2)]

    for it in items[1:]:
        tx = _norm_header(idx_to_text.get(it["idx"], ""))
        if tx is None:
            tx = ""
        rx1, ry1, rx2, ry2 = it["rect"]
        gap = float(rx1) - float(cur_rect[2])
        if gap <= thr:
            # łączymy: dodaj spację i tekst, rozszerz prostokąt
            if cur_text and tx:
                cur_text = f"{cur_text} {tx}".strip()
            elif tx:
                cur_text = tx
            # aktualizacja rect
            cur_rect[0] = min(cur_rect[0], float(rx1))
            cur_rect[1] = min(cur_rect[1], float(ry1))
            cur_rect[2] = max(cur_rect[2], float(rx2))
            cur_rect[3] = max(cur_rect[3], float(ry2))
        else:
            # zakończ segment i zacznij nowy
            segments.append({"text": cur_text.strip(), "rect": tuple(cur_rect)})
            cur_text = tx
            cur_rect = [float(rx1), float(ry1), float(rx2), float(ry2)]

    # dodaj ostatni segment
    segments.append({"text": cur_text.strip(), "rect": tuple(cur_rect)})
    # odfiltruj puste
    segments = [s for s in segments if s.get("text")] 
    return segments


def _infer_columns_global(rows):
    # Zbierz środki X i szerokości, uśrednij próg klastrowania
    xs = []
    ws = []
    for r in rows:
        for it in r:
            x1, y1, x2, y2 = it["rect"]
            xc = (x1 + x2) / 2.0
            w = max(1.0, x2 - x1)
            xs.append(xc)
            ws.append(w)
    if not xs:
        return []
    med_w = _median(ws) or 1.0
    thr = max(6.0, med_w * 0.75)
    # Klastrowanie 1D po xs (sort + łączenie jeśli odstęp < thr)
    pairs = sorted([(x, i) for i, x in enumerate(xs)], key=lambda p: p[0])
    centers = []
    cluster = [pairs[0][0]]
    for val, _ in pairs[1:]:
        if (val - cluster[-1]) <= thr:
            cluster.append(val)
        else:
            centers.append(sum(cluster) / len(cluster))
            cluster = [val]
    centers.append(sum(cluster) / len(cluster))
    centers.sort()
    return centers


def _assign_columns(rows, centers):
    # Przypisz każdą komórkę do najbliższego centrum kolumny
    for r in rows:
        for it in r:
            x1, y1, x2, y2 = it["rect"]
            xc = (x1 + x2) / 2.0
            if not centers:
                it["col"] = 0
                continue
            nearest = min(range(len(centers)), key=lambda i: abs(centers[i] - xc))
            it["col"] = nearest
    return rows


def analyze_table_structure(boxes: List[List[List[float]]], texts: List[str], scores: List[float]):
    """
    Heurystyczne rozpoznanie wierszy, kolumn i nagłówków z wyników OCR.
    Zwraca słownik z polami: columns_count, header_rows, rows (lista wierszy z komórkami),
    columns (opcjonalne zakresy x dla kolumn).
    """
    if not boxes:
        return {
            "columns_count": 0,
            "header_rows": [],
            "rows": []
        }

    # Zbuduj elementy z tekstem
    poly_with_text = []
    for b, t, s in zip(boxes, texts, scores):
        x1, y1, x2, y2 = _box_rect(b)
        poly_with_text.append({
            "box": b,
            "rect": (x1, y1, x2, y2),
            "text": t if not isinstance(t, (list, tuple)) else t[0],
            "score": float(s)
        })

    # Grupowanie na wiersze
    rows = _cluster_rows_by_y([p["box"] for p in poly_with_text])
    # Wrzucenie tekstów do struktur wierszy (mapowanie po porządku sortowania)
    # Utwórz mapa: rect->(text,score,box). Posłużymy się dopasowaniem po współrzędnych.
    rect_map = {}
    for p in poly_with_text:
        rect_map[p["rect"]] = p

    # Zamień każdy element wiersza (it) na pełny rekord z tekstem
    for ri, r in enumerate(rows):
        for j, it in enumerate(r):
            full = rect_map.get(it["rect"], None)
            if full:
                it.update({"text": full["text"], "score": full["score"]})
            else:
                it.update({"text": "", "score": 0.0})

    # Kolumny globalne z centrów X
    centers = _infer_columns_global(rows)
    rows = _assign_columns(rows, centers)
    columns_count = max((it.get("col", 0) for r in rows for it in r), default=-1) + 1

    # Policz orientacyjne granice kolumn z rectów
    columns_ranges = []
    for ci in range(columns_count):
        xs1 = []
        xs2 = []
        for r in rows:
            for it in r:
                if it.get("col") == ci:
                    x1, y1, x2, y2 = it["rect"]
                    xs1.append(x1)
                    xs2.append(x2)
        if xs1 and xs2:
            columns_ranges.append({"x1": float(min(xs1)), "x2": float(max(xs2))})
        else:
            columns_ranges.append({"x1": None, "x2": None})

    # Heurystyka naglowkow: sprobuj wykryc na podstawie tresci (zamiast zawsze brac pierwsza linie)
    header_info = detect_header_row_from_ocr(
        boxes,
        texts,
        min_headers_in_line=HEADER_MIN_MATCH_IN_LINE,
        threshold=HEADER_SIM_THRESHOLD,
    )
    if header_info:
        merged = header_info.get("merged_from")
        if merged:
            header_rows = sorted({int(x) for x in merged})
        else:
            header_rows = [int(header_info["row_index"])]
    else:
        header_rows = [0] if rows else []

    # Zbuduj strukture serializowalna
    serial_rows = []
    for r in rows:
        serial_rows.append([
            {
                "col": int(it.get("col", 0)),
                "text": it.get("text", ""),
                "score": float(it.get("score", 0.0)),
                "box": it["box"],
            }
            for it in r
        ])

    return {
        "columns_count": int(columns_count),
        "columns": columns_ranges,
        "header_rows": header_rows,
        "header_info": header_info,
        "rows": serial_rows,
    }


def _norm_header(s: str) -> str:
    if s is None:
        return ""
    s2 = str(s)
    # usunięcie końcowych dwukropków i nadmiarowych spacji, małe litery
    s2 = s2.replace("\n", " ")
    s2 = " ".join(s2.split()).strip().rstrip(":")
    return s2


def _similarity(a: str, b: str) -> float:
    # prosta miara podobieństwa (difflib) po znormalizowaniu
    import difflib
    ra = _norm_header(a).lower()
    rb = _norm_header(b).lower()
    if not ra or not rb:
        return 0.0
    return difflib.SequenceMatcher(None, ra, rb).ratio()


def _choose_mapping_for_header(header_text: str, threshold: float = 0.74) -> str:
    # Najpierw dopasowanie dokładne po znormalizowanym tekście
    h_norm = _norm_header(header_text)
    # mapujemy po kluczu znormalizowanym
    norm_map = { _norm_header(k): v for k, v in HEADER_MAP.items() }
    if h_norm in norm_map:
        return norm_map[h_norm]
    # inaczej wybierz najbardziej podobny powyżej progu
    best_key = None
    best_score = 0.0
    for k in HEADER_MAP.keys():
        sc = _similarity(h_norm, k)
        if sc > best_score:
            best_score = sc
            best_key = k
    if best_key is not None and best_score >= threshold:
        return HEADER_MAP[best_key]
    return ""  # brak mapowania


def _set_by_path(d: Dict[str, Any], path: str, value: Any):
    if not path:
        return
    parts = path.split('.')
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    last = parts[-1]
    # Jeśli pole już istnieje i nie jest puste – dołączamy z odstępem
    if last in cur and isinstance(cur[last], str) and cur[last].strip() and isinstance(value, str):
        cur[last] = (cur[last] + " " + value).strip()
    else:
        cur[last] = value


# === Fallback: dobieranie par nagłówek → wartość po „wyglądzie” nagłówka ===
def _upper_ratio(s: str) -> float:
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return 0.0
    upper = sum(1 for ch in letters if ch.isupper())
    return float(upper) / float(len(letters))


def _x_overlap_ratio(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    left = max(ax1, bx1)
    right = min(ax2, bx2)
    ov = max(0.0, right - left)
    aw = max(1.0, ax2 - ax1)
    bw = max(1.0, bx2 - bx1)
    base = min(aw, bw)
    return ov / base


def _build_rows_and_segments(boxes: List[List[List[float]]], texts: List[str]):
    rows = _cluster_rows_by_y(boxes)
    idx_to_text: Dict[int, str] = {}
    for idx, (b, t) in enumerate(zip(boxes, texts)):
        txt = t[0] if isinstance(t, (list, tuple)) and t else (t if isinstance(t, str) else str(t))
        idx_to_text[idx] = txt
    rows_segments = []
    for r in rows:
        segs = _merge_close_words_in_row(r, idx_to_text)
        rows_segments.append(segs)
    return rows, rows_segments


def _best_header_mapping_and_score(text: str) -> Tuple[str, float, str]:
    # Zwróć (path, score, matched_key)
    best_key = None
    best_score = 0.0
    for k in HEADER_MAP.keys():
        sc = _similarity(text, k)
        if sc > best_score:
            best_score = sc
            best_key = k
    path = HEADER_MAP[best_key] if (best_key is not None and best_score >= HEADER_SIM_THRESHOLD) else ""
    return path, best_score, (best_key or "")


def extract_pairs_from_ocr(boxes: List[List[List[float]]], texts: List[str]) -> Dict[str, Any]:
    """
    Heurystyczne dopasowywanie par nagłówek→wartość na podstawie tego, co wygląda jak nagłówek.
    1) Buduje linie i łączy wyrazy w segmenty.
    2) Wybiera segmenty nagłówko‑podobne (fuzzy do HEADER_MAP lub duże uppercase, krótka fraza).
    3) Dla każdego nagłówka szuka wartości w tej samej linii (na prawo) lub w kolejnych liniach poniżej.
    Zwraca słownik: {
      'record': {...},
      'pairs_diag': [{header, path, value, how, header_score, matched_key}]
    }
    """
    result_record: Dict[str, Any] = {"metadata": {}}
    pairs_diag: List[Dict[str, Any]] = []

    if not boxes or not texts:
        return {"record": result_record, "pairs_diag": pairs_diag}

    rows, rows_segments = _build_rows_and_segments(boxes, texts)
    # Konwertuj także segmenty do rozszerzonych struktur z rect (x1,y1,x2,y2)

    def is_header_like(seg_text: str, score: float) -> bool:
        t = seg_text.strip()
        if not t:
            return False
        words = t.split()
        # dopuszczamy krótkie frazy (<=4 słowa) lub duże ratio uppercase
        if score >= HEADER_LIKE_SIM_THRESHOLD:
            return True
        if len(words) <= 4 and _upper_ratio(t) >= 0.6:
            return True
        return False

    used_values = set()  # zbiór id segmentów użytych jako wartości (by nie dublować)

    for ri, segs in enumerate(rows_segments):
        # Przygotuj indeksy segmentów w obrębie wiersza do potrzeb wewnętrznych identyfikatorów
        # Tworzymy lokalne ID jako (ri, si)
        for si, seg in enumerate(segs):
            txt = seg.get("text", "")
            rect = seg.get("rect")
            if not rect:
                continue
            path, sim_score, matched_key = _best_header_mapping_and_score(txt)
            if not is_header_like(txt, sim_score):
                continue

            # Szukaj wartości w tej samej linii: segment najbliżej na prawo
            x1, y1, x2, y2 = rect
            best_same_row = None
            best_gap = 1e9
            for sj, seg2 in enumerate(segs):
                if sj == si:
                    continue
                r2 = seg2.get("rect")
                if not r2:
                    continue
                gx = r2[0] - x2
                if 0 <= gx <= PAIR_SAME_ROW_GAP_PX:
                    if gx < best_gap:
                        best_gap = gx
                        best_same_row = (ri, sj, seg2)

            chosen = None
            how = None
            if best_same_row is not None:
                key = (best_same_row[0], best_same_row[1])
                if key not in used_values:
                    chosen = best_same_row[2]
                    how = "same_row_right"

            # Jeśli nie znaleziono, szukaj w kolejnych wierszach poniżej
            if chosen is None:
                header_rect = rect
                for rj in range(ri + 1, min(len(rows_segments), ri + 1 + PAIR_BELOW_MAX_ROWS)):
                    cand_best = None
                    cand_gap_y = 1e9
                    for sj, seg2 in enumerate(rows_segments[rj]):
                        r2 = seg2.get("rect")
                        if not r2:
                            continue
                        if _x_overlap_ratio(header_rect, r2) >= PAIR_X_OVERLAP_MIN_RATIO:
                            gap_y = r2[1] - header_rect[3]
                            if gap_y >= -2:  # dopuszczalnie mały negatywny overlap
                                if gap_y < cand_gap_y:
                                    cand_gap_y = gap_y
                                    cand_best = (rj, sj, seg2)
                    if cand_best is not None:
                        key = (cand_best[0], cand_best[1])
                        if key not in used_values:
                            chosen = cand_best[2]
                            how = "below_rows"
                            break

            if chosen is None:
                continue

            value_text = chosen.get("text", "").strip()
            if not value_text:
                continue

            # Jeśli mamy już ścieżkę – użyj; jeśli nie, spróbuj wyznaczyć z mniejszym progiem
            final_path = path
            if not final_path:
                maybe_path = _choose_mapping_for_header(txt, threshold=HEADER_LIKE_SIM_THRESHOLD)
                if maybe_path:
                    final_path = maybe_path

            if not final_path:
                # Brak mapowania do zdefiniowanego pola – pomiń (można rozważyć zapis do metadata.misc)
                continue

            _set_by_path(result_record, final_path, value_text)
            used_values.add((ri, si))  # oznacz nagłówek jako zużyty (niekonieczne, ale porządek)
            pairs_diag.append({
                "header": txt,
                "path": final_path,
                "value": value_text,
                "how": how,
                "header_score": float(sim_score),
                "matched_key": matched_key,
            })

    return {"record": result_record, "pairs_diag": pairs_diag}


# === Ekstrakcja wielu rekordów „od nagłówka do dołu” na podstawie segmentów OCR ===
def extract_records_from_header_down(
    boxes: List[List[List[float]]],
    texts: List[str],
    min_headers_in_line: int = HEADER_MIN_MATCH_IN_LINE,
    header_threshold: float = HEADER_SIM_THRESHOLD,
    assign_overlap_min: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Buduje rekordy dla wszystkich wierszy znajdujących się poniżej wiersza nagłówków.
    - Wiersz nagłówka: linia z >= min_headers_in_line segmentami dopasowanymi do HEADER_MAP
      (z użyciem progu header_threshold).
    - Kolumny: zakresy X segmentów w wierszu nagłówka, zmapowanych do ścieżek pól.
    - Wiersze danych: wszystkie kolejne linie; segmenty przypisywane do kolumn, jeśli
      pokrycie poziome (x-overlap) z kolumną >= assign_overlap_min; wartości z wielu
      segmentów w danej kolumnie są łączone spacją.
    Zwraca listę rekordów w wymaganym schemacie (z polem metadata jako dict).
    """
    items: List[Dict[str, Any]] = []
    if not boxes or not texts:
        return items

    # 1) Zbuduj linie i segmenty
    rows, rows_segments = _build_rows_and_segments(boxes, texts)
    if not rows_segments:
        return items

    # 2) Znajdź wiersz nagłówków i zmapuj segmenty do ścieżek
    header_row_idx = None
    header_cols = []  # lista dict: {path, rect, text}
    for ri, segs in enumerate(rows_segments):
        mapped = []
        for seg in segs:
            txt = seg.get("text", "")
            if not txt:
                continue
            path = _choose_mapping_for_header(txt, threshold=header_threshold)
            if path:
                mapped.append({"path": path, "rect": seg.get("rect"), "text": txt})
        if len(mapped) >= min_headers_in_line:
            header_row_idx = ri
            # Usuń ewentualne duplikaty ścieżek – zostaw pierwsze wystąpienie wg x1
            mapped.sort(key=lambda m: (m["rect"][0] if m.get("rect") else 0.0))
            seen = set()
            for m in mapped:
                p = m["path"]
                if p in seen:
                    continue
                seen.add(p)
                header_cols.append(m)
            break

    if header_row_idx is None or not header_cols:
        return items

    # 3) Iteruj po wierszach od header_row_idx+1 i buduj rekordy
    for ri in range(header_row_idx + 1, len(rows_segments)):
        segs = rows_segments[ri]
        record: Dict[str, Any] = {"metadata": {}}
        # zbierz wartości per kolumna
        col_values: Dict[str, List[str]] = {hc["path"]: [] for hc in header_cols}
        for seg in segs:
            r2 = seg.get("rect")
            txt = (seg.get("text") or "").strip()
            if not r2 or not txt:
                continue
            # znajdź najlepszą kolumnę po pokryciu X
            best_path = None
            best_overlap = 0.0
            for hc in header_cols:
                hr = hc.get("rect")
                if not hr:
                    continue
                ov = _x_overlap_ratio(hr, r2)
                if ov > best_overlap:
                    best_overlap = ov
                    best_path = hc["path"]
            if best_path and best_overlap >= assign_overlap_min:
                col_values[best_path].append(txt)

        # zapisz do rekordu
        for path, vals in col_values.items():
            if not vals:
                continue
            value = " ".join(vals).strip()
            if value:
                _set_by_path(record, path, value)

        # czy wiersz zawiera jakiekolwiek dane
        has_any = any(k != "metadata" for k in record.keys()) or any(record.get("metadata", {}).values())
        if has_any:
            items.append(record)

    return items


def _extract_header_per_column(structure: Dict[str, Any]) -> Dict[int, str]:
    rows = structure.get("rows", [])
    header_rows = structure.get("header_rows", []) or []
    header_indices = header_rows if header_rows else ([0] if rows else [])
    col_to_header: Dict[int, str] = {}
    if not header_indices:
        return col_to_header
    tmp: Dict[int, List[str]] = {}
    for idx in header_indices:
        if idx < 0 or idx >= len(rows):
            continue
        for cell in rows[idx]:
            ci = int(cell.get("col", 0))
            txt = _norm_header(cell.get("text", ""))
            if not txt:
                continue
            tmp.setdefault(ci, []).append(txt)
    for ci, parts in tmp.items():
        col_to_header[ci] = " ".join(parts).strip()
    return col_to_header
    header_cells = rows[header_idx]
    # Może występować wiele komórek w tej samej kolumnie – łączymy
    tmp: Dict[int, List[str]] = {}
    for cell in header_cells:
        ci = int(cell.get("col", 0))
        txt = _norm_header(cell.get("text", ""))
        if not txt:
            continue
        tmp.setdefault(ci, []).append(txt)
    for ci, parts in tmp.items():
        col_to_header[ci] = " ".join(parts).strip()
    return col_to_header


def _map_columns_to_fields(structure: Dict[str, Any]) -> Dict[int, str]:
    headers = _extract_header_per_column(structure)
    mapping: Dict[int, str] = {}
    for ci, htxt in headers.items():
        path = _choose_mapping_for_header(htxt)
        if path:
            mapping[ci] = path
    return mapping


def extract_items_from_structure(structure: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = structure.get("rows", [])
    if not rows:
        return []
    col_to_path = _map_columns_to_fields(structure)
    if not col_to_path:
        return []

    header_rows = set(structure.get("header_rows", []) or [])
    items: List[Dict[str, Any]] = []
    for ri, row in enumerate(rows):
        if ri in header_rows:
            continue
        # Zbierz teksty per kolumna w danym wierszu
        per_col: Dict[int, List[str]] = {}
        for cell in row:
            ci = int(cell.get("col", 0))
            txt = str(cell.get("text", "")).strip()
            if not txt:
                continue
            per_col.setdefault(ci, []).append(txt)

        record: Dict[str, Any] = {"metadata": {}}
        for ci, texts in per_col.items():
            if ci not in col_to_path:
                continue
            value = " ".join([t for t in texts if t]).strip()
            if not value:
                continue
            _set_by_path(record, col_to_path[ci], value)

        # Jeżeli rekord ma choć jedno pole poza pustym metadata, dodaj
        has_any = any(k != "metadata" for k in record.keys()) or any(record.get("metadata", {}).values())
        if has_any:
            items.append(record)
    return items


def detect_header_row_from_ocr(boxes: List[List[List[float]]], texts: List[str],
                               min_headers_in_line: int = HEADER_MIN_MATCH_IN_LINE,
                               threshold: float = 0.74):
    """
    Wyszukuje wiersz nagłówka na podstawie pełnych wyników OCR dla całego obrazu.
    Zasada: jeżeli w jednej linii (wierszu wg klastrowania Y) znajduje się co
    najmniej min_headers_in_line komórek, których tekst mapuje się (dokładnie lub
    fuzz) do znanych nagłówków z HEADER_MAP, to tę linię uznajemy za wiersz nagłówka.

    Zwraca słownik z informacją o wykrytym wierszu:
      {
        'row_index': int,
        'y_top': float,
        'y_bottom': float,
        'matched_headers': ["Nazwa", "Opis rzeczy", ...],
      }
    albo None, jeśli nie znaleziono.
    """
    if not boxes or not texts:
        return None

    # Zbuduj elementy pomocnicze z indeksami, aby móc połączyć je z tekstami
    rows_raw = _cluster_rows_by_y(boxes)
    if not rows_raw:
        return None
    # Połącz sąsiadujące wiersze nagłówka, gdy są bardzo blisko siebie (np. „Rzecz” + „znaleziona”)
    merged_rows = []
    for idx, r in enumerate(rows_raw):
        y_top = min(it["rect"][1] for it in r)
        y_bottom = max(it["rect"][3] for it in r)
        if merged_rows:
            gap = y_top - merged_rows[-1]["y_bottom"]
            # Łącz maksymalnie kilka pierwszych sąsiadujących linii (typowo 2–3) jeśli są bardzo blisko
            if gap <= HEADER_ROW_MERGE_GAP and len(merged_rows[-1]["source_rows"]) < 3:
                merged_rows[-1]["items"].extend(r)
                merged_rows[-1]["y_top"] = min(merged_rows[-1]["y_top"], y_top)
                merged_rows[-1]["y_bottom"] = max(merged_rows[-1]["y_bottom"], y_bottom)
                merged_rows[-1]["source_rows"].append(idx)
                continue
        merged_rows.append({
            "items": list(r),
            "y_top": float(y_top),
            "y_bottom": float(y_bottom),
            "source_rows": [idx],
        })
    rows = merged_rows

    # Mapowanie idx -> text (kolejność zipa odpowiada enumeracji w _cluster_rows_by_y)
    idx_to_text: Dict[int, str] = {}
    for idx, (b, t) in enumerate(zip(boxes, texts)):
        # t może być tuple/list – ujednolić do stringa
        txt = t[0] if isinstance(t, (list, tuple)) and t else (t if isinstance(t, str) else str(t))
        idx_to_text[idx] = txt

    def find_best(rows_wrap, thr, min_count):
        best_local = None
        best_count_local = 0
        for ri, rwrap in enumerate(rows_wrap):
            r = rwrap["items"]
            segments = _merge_close_words_in_row(r, idx_to_text)
            matched_texts = []
            for seg in segments:
                txt = seg.get("text", "")
                if not txt:
                    continue
                path = _choose_mapping_for_header(txt, threshold=thr)
                if path:
                    matched_texts.append(_norm_header(txt))
            if len(matched_texts) >= min_count:
                count = len(matched_texts)
                y_top = rwrap["y_top"]
                y_bottom = rwrap["y_bottom"]
                if count > best_count_local or (count == best_count_local and best_local and y_top < best_local["y_top"]) or (best_local is None):
                    best_local = {
                        "row_index": min(rwrap["source_rows"]) if rwrap.get("source_rows") else ri,
                        "y_top": float(y_top),
                        "y_bottom": float(y_bottom),
                        "matched_headers": matched_texts,
                        "merged_from": rwrap.get("source_rows"),
                    }
                    best_count_local = count
        return best_local

    best = find_best(rows, threshold, min_headers_in_line)
    if best is None:
        # Fallback: luźniejszy próg i wystarczy jeden dopasowany nagłówek (np. „Rzecz znaleziona”)
        best = find_best(rows, HEADER_LIKE_SIM_THRESHOLD, 1)

    return best


def save_ocr_table_json(save_dir: str, image_path: str, roi_rect, boxes, texts, scores, table_meta=None, structure=None, records=None):
    ensure_dir(save_dir)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(save_dir, f"{base}_ocr_table.json")
    data = {
        "input_path": image_path,
        "roi_rect": list(roi_rect),
        "roi_box": rect_to_box(roi_rect),
        "results": [
            {"box": box, "text": txt, "score": float(sc)}
            for box, txt, sc in zip(boxes, texts, scores)
        ]
    }
    if table_meta is not None:
        data["table_meta"] = table_meta
    if structure is not None:
        data["structure"] = structure
    if records is not None:
        data["records"] = records
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


def save_items_json(save_dir: str, image_path: str, items: List[Dict[str, Any]]):
    ensure_dir(save_dir)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(save_dir, f"{base}_items.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Wykryj tabelę i uruchom OCR tylko na obszarze tabeli.")
    parser.add_argument("-i", "--input", required=True,
                        help="URL strony (http/https) – wykona pełny screenshot, lub ścieżka do obrazu.")
    args = parser.parse_args()

    ensure_dir("output")

    # Przygotuj wejściowy obraz: screenshot lub obraz z dysku
    src = args.input
    if is_url(src):
        screenshot_path = os.path.join("output", "screenshot_ocr_table.png")
        capture_screenshot(src, screenshot_path)
        img_path = screenshot_path
    else:
        img_path = src
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Nie znaleziono obrazu: {img_path}")

    im = Image.open(img_path).convert("RGB")

    # Przygotuj OCR (będzie użyty także do heurystyki nagłówków)
    ocr = PaddleOCR(use_textline_orientation=True)

    # 1) Detekcja tabel (pipeline V2)
    table_pipeline = TableRecognitionPipelineV2()  # domyślnie CPU, większa przenośność
    table_polys = detect_table_polygons_v2(table_pipeline, img_path)

    if not table_polys:
        # Brak tabel – spróbuj wykryć wiersz nagłówków na pełnym obrazie.
        # Jeśli znajdziemy linię z >= HEADER_MIN_MATCH_IN_LINE dopasowań do HEADER_MAP,
        # przyjmujemy ROI od tej linii w dół (cała szerokość obrazu).
        full_boxes, full_texts, full_scores, _ = ocr_with_tiling_on_image(ocr, im, offset=(0, 0))
        header_info = detect_header_row_from_ocr(full_boxes, full_texts,
                                                 min_headers_in_line=HEADER_MIN_MATCH_IN_LINE,
                                                 threshold=0.74)
        if header_info:
            y_top = max(0, int(np.floor(header_info["y_top"] - HEADER_TOP_MARGIN)))
            roi = (0, y_top, im.size[0], im.size[1])
            table_meta = {
                "note": "ROI wyznaczony heurystycznie z wiersza nagłówków",
                "header_info": header_info,
            }
        else:
            # Fallback: cały obraz
            roi = (0, 0, im.size[0], im.size[1])
            table_meta = {
                "note": "Nie wykryto tabel ani wiersza nagłówków – użyto całego obrazu"
            }
    else:
        rects = [polygon_to_rect(p) for p in table_polys]
        roi = union_rect(rects)
        table_meta = {
            "detected_table_count": len(rects),
            "table_boxes": [rect_to_box(r) for r in rects]
        }

    # Zapisz crop ROI do wglądu
    x1, y1, x2, y2 = roi
    crop = im.crop((x1, y1, x2, y2))
    base = os.path.splitext(os.path.basename(img_path))[0]
    crop_path = os.path.join("output", f"{base}_table_roi.png")
    crop.save(crop_path)

    # 2) OCR tylko na ROI (z przesunięciem do globalnych współrzędnych)
    boxes, texts, scores, _ = ocr_with_tiling_on_image(ocr, crop, offset=(x1, y1))

    # 3) Analiza struktury: wiersze, kolumny, nagłówki
    structure = analyze_table_structure(boxes, texts, scores)
    # 3b) Ekstrakcja rekordów wg mapy nagłówków + podobieństwo
    records = extract_items_from_structure(structure)

    # 3c) Jeśli brak, spróbuj trybu „od wiersza nagłówków w dół” w oparciu o segmenty OCR
    if not records:
        header_down_records = extract_records_from_header_down(
            boxes, texts,
            min_headers_in_line=HEADER_MIN_MATCH_IN_LINE,
            header_threshold=HEADER_SIM_THRESHOLD,
            assign_overlap_min=0.25,
        )
        if header_down_records:
            records = header_down_records
            if isinstance(table_meta, dict):
                table_meta.setdefault("notes", []).append(
                    "Zastosowano tryb rekonstrukcji rekordów: od wiersza nagłówków do dołu"
                )

    # 3d) Fallback: jeśli nadal nie udało się zbudować rekordów z tabeli,
    # spróbuj dobrać pary nagłówek→wartość na podstawie „wyglądu” nagłówków
    pairs_diag = None
    if not records:
        pairs_res = extract_pairs_from_ocr(boxes, texts)
        candidate = pairs_res.get("record", {}) or {"metadata": {}}
        # sprawdź czy ma jakiekolwiek pole poza pustym metadata
        has_any = any(k != "metadata" for k in candidate.keys()) or any(candidate.get("metadata", {}).values())
        if has_any:
            records = [candidate]
            pairs_diag = pairs_res.get("pairs_diag")
            # dopisz notatkę o fallbacku
            if isinstance(table_meta, dict):
                table_meta.setdefault("notes", []).append("Zastosowano fallback par nagłówek→wartość (poza tabelą)")
                if pairs_diag:
                    table_meta["pairs_diag"] = pairs_diag

    # 4) Zapis wyników i wizualizacja
    json_path = save_ocr_table_json("output", img_path, roi, boxes, texts, scores, table_meta, structure, records)
    items_path = save_items_json("output", img_path, records)

    # Wizualizacja: rysuj zielony prostokąt ROI i czerwone boxy OCR
    vis = draw_boxes_on_image(im, [rect_to_box(roi)], color=(0, 255, 0), width=3)
    vis = draw_boxes_on_image(vis, boxes, color=(255, 0, 0), width=2)
    vis_path = os.path.join("output", f"vis_{base}_ocr_table.png")
    vis.save(vis_path)

    print(f"ROI zapisano: {crop_path}")
    print(f"Wyniki OCR zapisano: {json_path}")
    print(f"Zbiór elementów zapisano: {items_path}")
    print(f"Wizualizację zapisano: {vis_path}")


if __name__ == "__main__":
    main()
