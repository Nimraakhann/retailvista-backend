from django.contrib import admin
from .models import UserProfile

# Register your models here.

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'store_id', 'title', 'company', 'email_verified')
    search_fields = ('user__email', 'user__first_name', 'user__last_name', 'company', 'store_id')
    list_filter = ('email_verified',)
    readonly_fields = ('store_id',)
