### Navigate to projekt-elegancki/backend

### Create and activate python environment

```
python3 -m venv venv
```

\+

If on Windows (CMD)
```
venv\Scripts\activate.bat
```

Else if on macOS
```
source venv/bin/activate
```

### Install mysql

```
pip install django pymysql
```

### Apply migrations

Ensure you have running MySQL instance with database config same as in settings.py DATABASE section

```
python3 manage.py migrate
```

### Run server

```
python3 manage.py runserver
```

### Example request to HTTPOST endpoint /form-post 

```
curl -X POST -H "Content-Type: application/json" -d '{"item": "kurtka", "foundDateTime": "2025-06-12", "location": "autobus 152" }' http://127.0.0.1:8000/form-post/
```