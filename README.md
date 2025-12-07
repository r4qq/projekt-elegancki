## Instalacja i Uruchomienie

Wymagany jest zainstalowany **Docker** oraz **Docker Compose**.

1.  **Sklonuj repozytorium** (jeśli jeszcze tego nie zrobiłeś).
2.  **Przejdź do głównego katalogu**:
    ```bash
    cd projekt-elegancki
    ```
3.  **Uruchom projekt**:
    ```bash
    docker-compose up --build
    ```
    *Flaga `--build` jest zalecana przy pierwszym uruchomieniu lub po zmianach w kodzie Python.*

Po uruchomieniu aplikacja jest dostępna pod adresem:
**http://localhost:80** (lub po prostu `http://localhost`)
