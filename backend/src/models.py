from django.db import models
import uuid
    
class LostItem(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    item = models.CharField(max_length=255)
    found_datetime = models.DateTimeField()
    location = models.CharField(max_length=255)
    metadata = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return self.item
    
