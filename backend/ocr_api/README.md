# OCR API (Django + Gunicorn)

Uruchamia endpoint `POST /api/ocr-table/` zwracający JSON jak `output/*_items.json` z `ocr_table.py`.

Endpoint akceptuje teraz kilka prostych sposobów przekazania obrazu:
- multipart/form-data: pole pliku o nazwie `image`, `file`, `photo` lub `upload`
- JSON: pole `image_base64` (czysty base64 lub data URL) lub `url`
- surowe body obrazka z nagłówkiem `Content-Type: image/*`

## Budowanie i uruchamianie (Docker)

```bash
docker build --no-cache -f ocr_api/Dockerfile -t ocr-api-local .
docker run --rm -p 8888:8000 --name ocr-api ocr-api-local
```

Port hosta można zmienić, np. `-p 9000:8000`.

## Wywołanie (PowerShell)

```powershell
Invoke-WebRequest -Uri "http://localhost:8888/api/ocr-table/" `
  -Method Post `
  -Form @{ image = Get-Item "table.png" } `
  -TimeoutSec 900 `
  -OutFile "response.json"

Get-Content response.json
```

### PowerShell: alternatywne nazwy pól
```powershell
Invoke-WebRequest -Uri "http://localhost:8888/api/ocr-table/" `
  -Method Post `
  -Form @{ file = Get-Item "table.png" } `
  -OutFile "response.json"
```

## Wywołanie (Node.js, fetch)

```js
import fs from "fs";
import fetch, { FormData, fileFromSync } from "node-fetch"; // lub native fetch w Node 18+ + undici

const form = new FormData();
form.append("image", fileFromSync("table.png"));

const res = await fetch("http://localhost:8888/api/ocr-table/", {
  method: "POST",
  body: form,
  // czas może być długi przy pierwszym ładowaniu modeli
});

if (!res.ok) {
  throw new Error(`HTTP ${res.status}`);
}

const data = await res.json();
console.log(data);
```

### JSON (base64)
```bash
BASE64=$(base64 -w0 table.png) # na Windows: certutil -encodehex -f table.png 12 | tr -d '\n' (lub narzędzie do base64)
curl -X POST http://localhost:8888/api/ocr-table/ \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$BASE64\"}"
```

### JSON (data URL)
```bash
curl -X POST http://localhost:8888/api/ocr-table/ \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"data:image/png;base64,AAA...\"}"
```

### URL do obrazka
```bash
curl -X POST http://localhost:8888/api/ocr-table/ \
  -F url=https://example.com/sample.png
```

### Surowy obraz (body)
```bash
curl -X POST http://localhost:8888/api/ocr-table/ \
  -H "Content-Type: image/png" \
  --data-binary @table.png
```

## Dane wejściowe
- multipart: `image`/`file`/`photo`/`upload` (plik) lub `url`
- JSON: `image_base64` (base64 lub data URL) lub `url`
- surowe body obrazu (Content-Type: image/*)

## Wynik
Lista rekordów jak w plikach `output/*_items.json` generowanych przez `ocr_table.py`.
