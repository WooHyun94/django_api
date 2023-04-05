from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils.translation import gettext_lazy as _


# Create your models here.
class User(AbstractUser):
    class Meta:
        db_table = 'users_tb'
        verbose_name = _('사용자')
        verbose_name_plural = _('사용자')