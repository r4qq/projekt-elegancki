import json
import base64
import mimetypes
import subprocess
import sys
import tempfile
from pathlib import Path
import time

import requests
from html.parser import HTMLParser
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt


def _write_upload_to_tmp(upload, tmp_dir: Path) -> Path:
    suffix = Path(upload.name).suffix or ".png"
    tmp_path = tmp_dir / f"input{suffix}"
    with tmp_path.open("wb") as f:
        for chunk in upload.chunks():
            f.write(chunk)
    return tmp_path


def _write_bytes_to_tmp(content: bytes, tmp_dir: Path, content_type: str | None = None) -> Path:
    """
    Zapisuje surowe bajty obrazu do pliku tymczasowego, ustalając rozszerzenie z nagłówka
    Content-Type jeśli dostępny.
    """
    ext = ".png"
    if content_type:
        ctype = content_type.lower()
        guessed = mimetypes.guess_extension(ctype)
        if guessed:
            ext = guessed
        elif "jpeg" in ctype:
            ext = ".jpg"
        elif "png" in ctype:
            ext = ".png"
    tmp_path = tmp_dir / f"input{ext}"
    tmp_path.write_bytes(content)
    return tmp_path


def _download_image(url: str, tmp_dir: Path) -> Path:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "").lower()

    # Jeżeli to strona HTML, spróbujmy zrobić screenshot całej strony.
    if "text/html" in ctype or (not ctype and resp.text.strip().lower().startswith("<!doctype html")):
        # Najpierw spróbujmy screenshotem (selenium). Jeśli się nie uda, spróbujemy wyłuskać obrazek z HTML.
        try:
            return _screenshot_webpage(url, tmp_dir)
        except Exception:
            # Fallback: spróbuj wyciągnąć obraz (og:image albo pierwszy <img>)
            try:
                return _download_image_from_html(resp.text, url, tmp_dir)
            except Exception as e:
                raise ValueError("URL nie wskazuje bezpośrednio na obraz i nie udało się wykonać zrzutu ekranu ani znaleźć obrazka na stronie.") from e

    # Jeśli to obraz
    if ctype.startswith("image/"):
        ext = mimetypes.guess_extension(ctype) or (".jpg" if "jpeg" in ctype else ".png")
        tmp_path = tmp_dir / f"input{ext}"
        tmp_path.write_bytes(resp.content)
        return tmp_path

    # Nie rozpoznano typu — spróbuj jako obraz albo odrzuć
    if not ctype:
        # Zgadnij po rozszerzeniu
        guessed_ext = Path(url).suffix.lower()
        if guessed_ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
            tmp_path = tmp_dir / f"input{guessed_ext}"
            tmp_path.write_bytes(resp.content)
            return tmp_path

    raise ValueError("URL nie wskazuje na obsługiwany obraz ani stronę HTML.")


class _ImageFromHtmlParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.og_image = None
        self.first_img = None

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "meta":
            attr = dict(attrs)
            prop = attr.get("property") or attr.get("name")
            if prop and prop.lower() in {"og:image", "twitter:image"}:
                content = attr.get("content")
                if content and not self.og_image:
                    self.og_image = content
        elif tag.lower() == "img" and self.first_img is None:
            attr = dict(attrs)
            src = attr.get("src") or attr.get("data-src")
            if src:
                self.first_img = src


def _download_image_from_html(html: str, base_url: str, tmp_dir: Path) -> Path:
    parser = _ImageFromHtmlParser()
    parser.feed(html)
    img_url = parser.og_image or parser.first_img
    if not img_url:
        raise ValueError("Nie znaleziono obrazka na stronie (og:image / <img>).")
    img_url_abs = urljoin(base_url, img_url)
    return _download_image(img_url_abs, tmp_dir)


def _get_chrome_driver(options: Options) -> webdriver.Chrome:
    """
    Zwraca instancję webdriver.Chrome. Najpierw próbuje użyć systemowego chromedrivera
    (np. /usr/bin/chromedriver), a jeśli się nie powiedzie, używa webdriver-manager.
    """
    # Spróbuj wskazać binary Chromium/Chrome jeśli dostępny
    for bin_path in ("/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome", "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"):
        try:
            if Path(bin_path).exists():
                options.binary_location = bin_path
                break
        except Exception:
            pass

    # Spróbuj systemowego chromedrivera
    for cand in ("/usr/bin/chromedriver", "C:\\Program Files\\Google\\Chrome\\Application\\chromedriver.exe"):
        try:
            service = Service(cand)
            return webdriver.Chrome(service=service, options=options)
        except Exception:
            pass
    # Fallback: webdriver-manager (bez ChromeType – kompatybilne z aktualnymi wersjami)
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def _screenshot_webpage(url: str, tmp_dir: Path, viewport_width: int = 1366, init_wait_ms: int = 1000) -> Path:
    """
    Tworzy zrzut całej strony za pomocą Chrome w trybie headless. Wymaga zainstalowanego Chrome/Chromium.
    Jeśli Chrome nie jest dostępny, funkcja podniesie wyjątek.
    """
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--hide-scrollbars")
    options.add_argument(f"--window-size={viewport_width},1200")

    driver = _get_chrome_driver(options)
    try:
        driver.get(url)
        if init_wait_ms:
            time.sleep(init_wait_ms / 1000.0)

        # Ustal pełną wysokość dokumentu
        total_height = driver.execute_script(
            "return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight, document.body.offsetHeight, document.documentElement.offsetHeight, document.body.clientHeight, document.documentElement.clientHeight);"
        )
        # Zmień rozmiar okna na pełną wysokość dokumentu (Chrome potrafi robić pełny screenshot)
        driver.set_window_size(viewport_width, max(1200, int(total_height)))
        png = driver.get_screenshot_as_png()
        tmp_path = tmp_dir / "input.png"
        tmp_path.write_bytes(png)
        return tmp_path
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def _decode_base64_image_to_tmp(b64data: str, tmp_dir: Path) -> Path:
    """
    Akceptuje zarówno czysty base64, jak i data URL (data:image/png;base64,XXXX).
    """
    content_type = None
    if b64data.strip().startswith("data:"):
        header, b64 = b64data.split(",", 1)
        # przykład: data:image/png;base64
        if ";base64" in header:
            content_type = header.split(";")[0][5:]
        b64data = b64
    try:
        raw = base64.b64decode(b64data, validate=True)
    except Exception:
        # Spróbuj bez validate (częste w praktyce)
        raw = base64.b64decode(b64data)
    return _write_bytes_to_tmp(raw, tmp_dir, content_type)


def _run_ocr_table(image_path: Path) -> Path:
    """
    Uruchamia istniejący skrypt ocr_table.py i zwraca ścieżkę do wygenerowanego pliku items JSON.
    """
    base = image_path.stem
    output_items = settings.BASE_DIR / "output" / f"{base}_items.json"
    cmd = [sys.executable, str(settings.BASE_DIR / "ocr_table.py"), "-i", str(image_path)]
    subprocess.run(cmd, check=True, cwd=settings.BASE_DIR)
    if not output_items.exists():
        raise FileNotFoundError(f"Nie znaleziono {output_items}")
    return output_items


@csrf_exempt
def ocr_table_view(request):
    if request.method != "POST":
        return JsonResponse({"detail": "Only POST is allowed"}, status=405)

    # 1) Multipart/form-data: zaakceptuj kilka popularnych nazw pól
    upload = (
        request.FILES.get("image")
        or request.FILES.get("file")
        or request.FILES.get("photo")
        or request.FILES.get("upload")
    )

    # 2) URL może przyjść w multipart albo w JSON
    url = request.POST.get("url")

    # 3) JSON body: image_base64 lub url
    json_body = None
    if request.content_type and "application/json" in request.content_type:
        try:
            json_body = json.loads(request.body.decode("utf-8")) if request.body else {}
        except Exception:
            return HttpResponseBadRequest("Invalid JSON body.")
        if isinstance(json_body, dict):
            url = url or json_body.get("url")
            image_b64 = json_body.get("image_base64")
        else:
            image_b64 = None
    else:
        image_b64 = None

    # 4) Surowe bajty obrazu (np. Content-Type: image/png) jeśli nie ma multipart/json
    is_raw_image = (
        not upload
        and not url
        and not image_b64
        and request.content_type
        and request.content_type.lower().startswith("image/")
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            if upload:
                img_path = _write_upload_to_tmp(upload, tmp_dir)
            elif image_b64:
                img_path = _decode_base64_image_to_tmp(image_b64, tmp_dir)
            elif url:
                img_path = _download_image(url, tmp_dir)
            elif is_raw_image:
                img_path = _write_bytes_to_tmp(request.body, tmp_dir, request.content_type)
            else:
                return HttpResponseBadRequest(
                    "Provide an image via one of: multipart file (image/file/photo/upload), JSON 'image_base64', raw image body (Content-Type: image/*), or 'url'."
                )

            items_path = _run_ocr_table(img_path)
            data = json.loads(items_path.read_text(encoding="utf-8"))
            return JsonResponse(data, safe=False)
        except ValueError as exc:
            return HttpResponseBadRequest(str(exc))
        except Exception as exc:
            return JsonResponse({"detail": str(exc)}, status=500)
