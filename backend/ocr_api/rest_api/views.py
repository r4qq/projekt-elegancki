import json
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


def _download_image(url: str, tmp_dir: Path) -> Path:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "").lower()
    ext = ".jpg" if "jpeg" in ctype else ".png"
    tmp_path = tmp_dir / f"input{ext}"
    tmp_path.write_bytes(resp.content)
    return tmp_path


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

    url = request.POST.get("url")
    upload = request.FILES.get("image")
    if not url and not upload:
        return HttpResponseBadRequest("Provide either 'url' field or 'image' file.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            if upload:
                img_path = _write_upload_to_tmp(upload, tmp_dir)
            else:
                img_path = _download_image(url, tmp_dir)

            items_path = _run_ocr_table(img_path)
            data = json.loads(items_path.read_text(encoding="utf-8"))
            return JsonResponse(data, safe=False)
        except Exception as exc:
            return JsonResponse({"detail": str(exc)}, status=500)
