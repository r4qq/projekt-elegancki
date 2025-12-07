from django.urls import path
from .views import ocr_table_view

urlpatterns = [
    path("ocr-table/", ocr_table_view, name="ocr-table"),
]
