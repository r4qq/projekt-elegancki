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
        found_datetime_str = data.get('foundDateTime') or data.get('found_datetime')
        location = data.get('location')
        metadata = data.get('metadata') or {}

        # Domknięcie metadanych minimalnym zestawem gdy brak
        now_iso = timezone.now().isoformat()
        metadata.setdefault('createdAt', now_iso)
        metadata.setdefault('enteredAt', now_iso)

        parsed_dt = parse_datetime(found_datetime_str) if found_datetime_str else None
        if parsed_dt is None:
            parsed_dt = timezone.now()

        item = LostItem.objects.create(
            item=item,
            found_datetime=parsed_dt,
            location=location,
            metadata=metadata
        )

        return JsonResponse({
            'message': 'Przedmiot dodany pomyślnie',
            'id': item.id
        }, status=201)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


def list_lost_items(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Dozwolona tylko metoda GET'}, status=405)

    items = LostItem.objects.all().order_by('-found_datetime')
    data = []
    for obj in items:
        data.append({
            'id': str(obj.id),
            'item': obj.item,
            'foundDateTime': obj.found_datetime.isoformat(),
            'location': obj.location,
            'metadata': obj.metadata or {}
        })
    return JsonResponse(data, safe=False, status=200)
    
