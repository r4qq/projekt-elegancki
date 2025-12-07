# OCR API (Django + Gunicorn)

Uruchamia endpoint `POST /api/ocr-table/` zwracający JSON jak `output/*_items.json` z `ocr_table.py`.

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

## Dane wejściowe
- `image` (multipart/form-data) **lub** `url` (bezpośredni URL do pliku graficznego).

## Wynik
Lista rekordów jak w plikach `output/*_items.json` generowanych przez `ocr_table.py`.
