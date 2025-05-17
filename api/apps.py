from django.apps import AppConfig
import threading
import time
from django.utils import timezone


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    
    def ready(self):
        # Don't run cleanup thread in Django's development auto-reloader
        import sys
        if not ('runserver' in sys.argv or 'runserver_plus' in sys.argv) or ('--noreload' in sys.argv):
            # Start a background thread that runs the cleanup task periodically
            cleanup_thread = threading.Thread(target=self._cleanup_task, daemon=True)
            cleanup_thread.start()
    
    def _cleanup_task(self):
        """Run cleanup tasks periodically"""
        # Wait for Django to fully initialize
        time.sleep(10)
        
        # Import inside the method to avoid circular imports
        from .models import ShopliftingAlert
        
        while True:
            try:
                print(f"[{timezone.now()}] Running scheduled cleanup of old alerts...")
                ShopliftingAlert.cleanup_old_alerts()
                print(f"[{timezone.now()}] Cleanup completed")
            except Exception as e:
                print(f"Error in cleanup task: {e}")
            
            # Sleep for 1 hour between cleanups
            time.sleep(3600)
