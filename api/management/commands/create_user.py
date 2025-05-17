from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password

class Command(BaseCommand):
    help = 'Create a new superuser'

    def handle(self, *args, **kwargs):
        email = 'admin@retailvista.com'
        username = 'admin'
        password = 'admin123'
        
        try:
            # Delete existing user if exists
            User.objects.filter(email=email).delete()
            
            # Create new superuser with properly hashed password
            user = User.objects.create(
                username=username,
                email=email,
                password=make_password(password),
                first_name='Admin',
                last_name='User',
                is_staff=True,
                is_superuser=True
            )
            
            self.stdout.write(self.style.SUCCESS(f'Successfully created superuser: {user.email}'))
            self.stdout.write('Login credentials:')
            self.stdout.write(f'Email: {email}')
            self.stdout.write(f'Password: {password}')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error creating user: {str(e)}'))
