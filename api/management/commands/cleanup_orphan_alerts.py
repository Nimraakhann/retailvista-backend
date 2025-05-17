from django.core.management.base import BaseCommand
from api.models import ShopliftingAlert, Camera

class Command(BaseCommand):
    help = 'Delete all shoplifting alerts for cameras that do not exist or are not active.'

    def handle(self, *args, **options):
        orphan_alerts = ShopliftingAlert.objects.exclude(camera__is_active=True)
        count = orphan_alerts.count()
        orphan_alerts.delete()
        self.stdout.write(self.style.SUCCESS(f'Deleted {count} orphaned shoplifting alerts.')) 