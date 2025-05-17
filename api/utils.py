from django.core.mail import send_mail
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

def validate_password(password):
    """
    Validate password meets requirements:
    - At least 8 characters long
    - Contains at least one uppercase letter
    - Contains at least one number
    - Contains at least one special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    
    if not any(c in "!@#$%^&*" for c in password):
        return False, "Password must contain at least one special character (!@#$%^&*)"
    
    return True, "Password is valid"