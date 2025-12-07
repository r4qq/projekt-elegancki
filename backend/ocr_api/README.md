# OCR API (Django + Gunicorn)

Uruchamia endpoint `POST /api/ocr-table/` zwracający JSON jak `output/*_items.json` z `ocr_table.py`.

Endpoint akceptuje teraz kilka prostych sposobów przekazania obrazu:
- multipart/form-data: pole pliku o nazwie `image`, `file`, `photo` lub `upload`
- JSON: pole `image_base64` (czysty base64 lub data URL) lub `url`
- surowe body obrazka z nagłówkiem `Content-Type: image/*`

Jeśli przekażesz `url` wskazujący NA STRONĘ WWW (HTML), API spróbuje:
1) wykonać zrzut ekranu całej strony (Chrome/Chromium headless), a następnie ten screenshot przetworzyć OCR,
2) gdy screenshot się nie uda, pobrać obraz znaleziony w HTML (meta `og:image` lub pierwszy `<img>`),
3) jeśli nadal nie ma obrazu — zwróci błąd 400 z informacją, że nie znaleziono obrazu.

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

### Node.js: wysłanie URL (JSON)
```js
import fetch from "node-fetch"; // w Node 18+ możesz użyć wbudowanego fetch (lub undici)

const res = await fetch("http://localhost:8888/api/ocr-table/", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    // może być URL do obrazka lub do strony WWW (zrzut ekranu)
    url: "https://example.com/strona-z-tabela",
  }),
});

if (!res.ok) {
  throw new Error(`HTTP ${res.status}`);
}
const data = await res.json();
console.log(data);
```

### Node.js: wysłanie URL (FormData)
```js
import fetch, { FormData } from "node-fetch"; // lub wbudowany fetch + undici

const form = new FormData();
form.append("url", "https://example.com/sample.png");

const res = await fetch("http://localhost:8888/api/ocr-table/", {
  method: "POST",
  body: form,
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

### URL do strony (screenshot całej strony)
```bash
curl -X POST http://localhost:8888/api/ocr-table/ \
  -F url=https://example.com/strona-z-tabela
```
Uwagi:
- API zrobi pełny zrzut strony (bez przewijania po kawałku) w trybie headless i użyje go dla OCR.
- Jeśli zrzut nie jest możliwy (brak Chromium), API spróbuje obrazka `og:image` lub pierwszego `<img>`.
- Serwis nie wykonuje logowania i nie obsługuje stron wymagających interakcji; screenshot jest po renderze klienta, ale bez autoryzacji.

### Przeglądarka (fetch) — wysłanie URL
Uwaga: wywołania z przeglądarki wymagają CORS (serwer musi zwracać odpowiednie nagłówki) lub powinny być wykonywane przez Twój backend jako proxy. Poniższe przykłady działają bez CORS, jeśli wywołujesz z tej samej domeny/portu.

JSON:
```html
<script>
async function runJson() {
  const res = await fetch("/api/ocr-table/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url: "https://example.com/strona-z-tabela" })
  });
  if (!res.ok) throw new Error("HTTP " + res.status);
  const data = await res.json();
  console.log(data);
}
</script>
```

FormData:
```html
<script>
async function runForm() {
  const fd = new FormData();
  fd.append("url", "https://example.com/sample.png");
  const res = await fetch("/api/ocr-table/", { method: "POST", body: fd });
  if (!res.ok) throw new Error("HTTP " + res.status);
  const data = await res.json();
  console.log(data);
}
</script>
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

## Wymagania dla screenshotów stron
- Dockerfile instaluje `chromium` i `chromium-driver`, aby umożliwić zrzuty ekranu w kontenerze.
- W środowisku lokalnym (bez Dockera) zainstaluj Chrome/Chromium i chromedriver zgodny z wersją przeglądarki.

## Wynik
Lista rekordów jak w plikach `output/*_items.json` generowanych przez `ocr_table.py`.
