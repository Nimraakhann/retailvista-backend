from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from api.models import UserProfile

class Command(BaseCommand):
    help = 'Creates a test user'

    def handle(self, *args, **kwargs):
        try:
            # Check if test user exists
            if User.objects.filter(email='test@example.com').exists():
                self.stdout.write(self.style.WARNING('Test user already exists'))
                return

            # Create user
            user = User.objects.create_user(
                username='testuser',
                email='test@example.com',
                password='password123',
                first_name='Test',
                last_name='User'
            )

            # Create profile
            UserProfile.objects.create(
                user=user,
                title='Test Title',
                company='Test Company',
                phone='123-456-7890'
            )

            self.stdout.write(self.style.SUCCESS('Successfully created test user'))
            self.stdout.write('Email: test@example.com')
            self.stdout.write('Password: password123')

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error creating test user: {str(e)}'))
