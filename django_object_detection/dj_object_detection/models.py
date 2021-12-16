from django.db import models

# Create your models here.
class Image(models.Model):
    image= models.URLField(null=True)

    class Meta:
        db_table = 'image'
class image_storage(models.Model):
    id = models.CharField(primary_key=True,max_length=10)
    image_url= models.URLField(null=True)

    class Meta:
        db_table = 'image_storage'