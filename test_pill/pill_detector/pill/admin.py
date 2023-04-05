from django.contrib import admin

from pill.models import Pill


# Register your models here.
@admin.register(Pill)
class PillAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'created_at', 'updated_at']
    list_display_links = ['id', 'name']