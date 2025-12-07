import json
import base64
import mimetypes
import subprocess
import sys
import tempfile
from pathlib import Path

import requests
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
    ext = ".jpg" if "jpeg" in ctype else ".png"
    tmp_path = tmp_dir / f"input{ext}"
    tmp_path.write_bytes(resp.content)
    return tmp_path


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
        except Exception as exc:
            return JsonResponse({"detail": str(exc)}, status=500)
