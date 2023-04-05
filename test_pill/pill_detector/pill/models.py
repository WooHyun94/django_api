from django.db import models
from django.utils.translation import gettext_lazy as _

from core.models import HistoryModel


# Create your models here.
class Pill(HistoryModel):
    name = models.CharField(
        max_length=100,
        db_index=True, 
        verbose_name=_('이름')
    )

    efficacy = models.TextField(
        verbose_name=_('효과')
    )

    class Meta:
        ordering = ['-id']
        db_table = 'pill_tb'
        verbose_name = _('약')
        verbose_name_plural = _('약')