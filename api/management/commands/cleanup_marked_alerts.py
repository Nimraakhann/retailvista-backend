from django.core.management.base import BaseCommand
from django.utils import timezone
from api.models import ShopliftingAlert


class Command(BaseCommand):
    help = 'Cleanup all reviewed shoplifting alerts that might not have been deleted yet'

    def add_arguments(self, parser):
        parser.add_argument('--id', type=int, help='Specific alert ID to clean up')

    def handle(self, *args, **options):
        alert_id = options.get('id')
        
        if alert_id:
            # Clean up a specific alert
            try:
                alert = ShopliftingAlert.objects.get(id=alert_id)
                self.stdout.write(self.style.WARNING(f'Found alert with ID {alert_id}'))
                
                # Force delete the alert
                alert.delete()
                self.stdout.write(self.style.SUCCESS(f'Successfully deleted alert with ID {alert_id}'))
            except ShopliftingAlert.DoesNotExist:
                self.stdout.write(self.style.ERROR(f'Alert with ID {alert_id} not found'))
            return

        # Get all alerts that are marked as reviewed
        reviewed_alerts = ShopliftingAlert.objects.filter(is_reviewed=True)
        count = reviewed_alerts.count()
        
        self.stdout.write(self.style.WARNING(f'Found {count} reviewed alerts that were not deleted yet'))
        
        if count > 0:
            # Set auto_delete_date to now for all of them
            reviewed_alerts.update(auto_delete_date=timezone.now())
            
            # Run the cleanup
            ShopliftingAlert.cleanup_old_alerts()
            
            self.stdout.write(self.style.SUCCESS(f'Successfully deleted {count} reviewed alerts'))
        else:
            self.stdout.write(self.style.SUCCESS('No reviewed alerts need cleanup'))
            
        # List all remaining alerts
        remaining = ShopliftingAlert.objects.all()
        if remaining.count() > 0:
            self.stdout.write(self.style.WARNING(f'Remaining alerts in database:'))
            for alert in remaining:
                self.stdout.write(f'ID: {alert.id}, Reviewed: {alert.is_reviewed}, Auto-delete: {alert.auto_delete_date}')
        else:
            self.stdout.write(self.style.SUCCESS('No alerts remaining in database')) 