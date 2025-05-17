from django.core.management.base import BaseCommand
from api.models import Camera, ShopliftingAlert

class Command(BaseCommand):
    help = 'Delete a Camera and related alerts by camera_id.'

    def add_arguments(self, parser):
        parser.add_argument('camera_id', type=str, help='Camera ID to delete')

    def handle(self, *args, **options):
        camera_id = options['camera_id']
        try:
            camera = Camera.objects.get(camera_id=camera_id)
            alerts = ShopliftingAlert.objects.filter(camera=camera)
            alert_count = alerts.count()
            alerts.delete()
            camera.delete()
            self.stdout.write(self.style.SUCCESS(f'Deleted camera {camera_id} and {alert_count} related alerts.'))
        except Camera.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'Camera with camera_id {camera_id} does not exist.')) 