# Specyfikacja Obiektu JSON: Zgłoszenie Znalezionego Przedmiotu

## 1. Opis ogólny
Obiekt reprezentuje rekord w systemie "Biura Rzeczy Znalezionych". Przechowuje informacje o fizycznym przedmiocie, lokalizacji i czasie jego znalezienia, a także metadane administracyjne dotyczące procesu rejestracji zgłoszenia.

## 2. Struktura głównego obiektu

Poniższa tabela opisuje pola znajdujące się w głównym korzeniu (root) obiektu JSON.

| Pole | Typ danych | Wymagalność | Opis | Format / Przykład |
| :--- | :--- | :---: | :--- | :--- |
| `item` | `string` | **Wymagane** | Krótki opis lub nazwa znalezionego przedmiotu. | "Telefon komórkowy marki SAMSUNG" |
| `foundDateTime` | `string` | **Wymagane** | Data i godzina znalezienia przedmiotu. Zgodna ze standardem ISO 8601. | `YYYY-MM-DDTHH:mm:ss` <br> np. "2025-03-31T14:30:00" |
| `location` | `string` | **Wymagane** | Opis miejsca, w którym przedmiot został znaleziony. Może zawierać miasto, ulicę oraz charakterystyczny punkt. | "Poznań, Plac Wilsona, ławka przy fontannie" |
| `metadata` | `object` | **Wymagane** | Obiekt zawierający informacje administracyjne o rekordzie (kto i kiedy go utworzył). | Patrz sekcja **3. Obiekt Metadata** |

## 3. Obiekt Metadata (`metadata`)

Zagnieżdżony obiekt przechowujący logi systemowe i informacje o autorstwie wpisu.

| Pole | Typ danych | Opis | Format / Uwagi |
| :--- | :--- | :--- | :--- |
| `createdBy` | `string` | Imię i nazwisko osoby, która fizycznie zgłosiła lub jest właścicielem zgłoszenia (np. znalazca lub dyspozytor). | "Karolina Jakubowska" |
| `createdAt` | `string` | Data utworzenia zgłoszenia (data wpłynięcia informacji). | Format daty: `YYYY-MM-DD` |
| `enteredBy` | `string` | Imię i nazwisko pracownika/operatora systemu, który fizycznie wprowadził dane do bazy. | "Góral Aleksander" |
| `enteredAt` | `string` | Data wprowadzenia rekordu do bazy danych. | Format daty: `YYYY-MM-DD` |

## 4. Przykładowy Payload

Poniżej znajduje się wzorcowy obiekt JSON zgodny z powyższą specyfikacją.

```json
{
  "item": "Telefon komórkowy marki SAMSUNG",
  "foundDateTime": "2025-03-31T14:30:00",
  "location": "Poznań, Plac Wilsona, ławka przy fontannie",
  "metadata": {
    "createdBy": "Karolina Jakubowska",
    "createdAt": "2025-03-31",
    "enteredBy": "Góral Aleksander",
    "enteredAt": "2025-03-31"
  }
}