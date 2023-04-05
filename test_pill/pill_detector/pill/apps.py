from django.apps import AppConfig
from django.utils.translation import gettext_lazy as _


class PillConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'pill'
    verbose_name = _('약')