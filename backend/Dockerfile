FROM python:3.11-slim

#Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

#Work directory
WORKDIR /app

#System dependencies required for mysqlclient
RUN apt-get update && apt-get install -y \
    pkg-config \
    gcc \
    curl \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

#Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

#Copy project
COPY . /app/

#Expose the port
EXPOSE 8000

#Run database migrations and start server
CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]