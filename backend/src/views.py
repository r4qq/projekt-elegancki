import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.dateparse import parse_datetime
from django.utils import timezone
from .models import LostItem

@csrf_exempt
def add_lost_item(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Dozwolona tylko metoda POST'}, status=405)

    try:
        data = json.loads(request.body)
        
        # Pobieranie danych z JSON (według nazw z formularza frontendowego)
        item = data.get('item')
        found_datetime_str = data.get('foundDateTime')
        location = data.get('location')
        
        now = timezone.now().isoformat(),
        metadata = {
            'createdBy': 'Jan Kowalski',
            'createdAt': now,
            'enteredBy': 'Piotr Nowak',
            'enteredAt': now
        }

        # Tworzenie obiektu
        item = LostItem.objects.create(
            item=item,
            found_datetime=parse_datetime(found_datetime_str), # Konwersja stringa na datę
            location=location,
            metadata=metadata
        )

        return JsonResponse({
            'message': 'Przedmiot dodany pomyślnie',
            'id': item.id
        }, status=201)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
    