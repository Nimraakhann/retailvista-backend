import os
import base64
import cv2
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.contrib.auth.models import User
from .models import UserProfile, Camera, ShopliftingAlert
from .utils import validate_password
from django.db import transaction
from django.core.mail import send_mail
from django.contrib.auth import get_user_model
from django.conf import settings
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
import logging
from .models import Promotion
from .serializers import PromotionSerializer
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from django.core.mail import send_mail
from django.conf import settings
from django.utils import timezone
from datetime import timedelta
import json
import uuid
import numpy as np
import threading
from threading import Lock
from collections import defaultdict
import time

from .models import UserProfile, Map
from .ml_models.shoplift.detect import ShopliftDetector
from .ml_models.age_gender.age_gen import AgeGenderDetector
from .ml_models.people_counter.PeopleCounterDetector import PeopleCounterDetector
from .models import DetectionData
from django.db.models import Sum, Count
from django.db.models.functions import TruncHour, TruncDay, TruncWeek, TruncMonth

User = get_user_model()
logger = logging.getLogger(__name__)

detector_instances = {}
detector_lock = Lock()

age_gender_detector_instances = {}
age_gender_detector_lock = Lock()

people_counter_detectors = {}

@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def connect_camera(request):
    data = json.loads(request.body)
    video_path = data.get('video_path')
    camera_name = data.get('name', 'Camera')  # Get camera name if provided
    
    if not video_path:
        return JsonResponse({
            'status': 'error', 
            'message': 'Missing video_path'
        }, status=400)
    
    try:
        with detector_lock:
            # Generate a unique identifier for this camera
            camera_id = str(uuid.uuid4())
            
            # Create camera record in database
            camera = Camera.objects.create(
                user=request.user,
                camera_id=camera_id,
                video_path=video_path,
                camera_type='shoplift',
                name=camera_name
            )
            
            # Get auth token
            token = request.META.get('HTTP_AUTHORIZATION', '').split(' ')[1] if 'HTTP_AUTHORIZATION' in request.META else None
            if not token:
                # If token not in header, generate a new one
                refresh = RefreshToken.for_user(request.user)
                token = str(refresh.access_token)
            
            # Initialize detector for this camera
            detector = ShopliftDetector()
            result = detector.start_detection(video_path, camera_id, token)
            
            if result:
                detector_instances[camera_id] = detector
                
                return JsonResponse({
                    'status': 'success', 
                    'camera_id': camera_id,
                    'message': 'Camera connected successfully'
                })
            else:
                # If detector fails to start, delete the camera record
                camera.delete()
                return JsonResponse({
                    'status': 'error',
                    'message': 'Failed to initialize detection'
                }, status=500)
    except Exception as e:
        print(f"Error connecting camera: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def connect_age_gender_camera(request):
    data = json.loads(request.body)
    camera_id = data.get('camera_id')
    video_path = data.get('video_path')
    name = data.get('name')  # Add this
    
    if not camera_id or not video_path:
        return JsonResponse({
            'status': 'error', 
            'message': 'Missing camera_id or video_path'
        })
    
    try:
        with age_gender_detector_lock:
            # Check for existing camera
            camera = Camera.objects.filter(
                camera_id=camera_id,
                user=request.user,
                camera_type='age_gender'
            ).first()
            
            if not camera:
                camera = Camera.objects.create(
                    user=request.user,
                    camera_id=camera_id,
                    video_path=video_path,
                    name=name,  # Add this
                    camera_type='age_gender'
                )
            
            # Start detector if not already running
            if camera_id not in age_gender_detector_instances:
                # Get the JWT token from the request
                auth_header = request.headers.get('Authorization')
                if auth_header and auth_header.startswith('Bearer '):
                    auth_token = auth_header.split(' ')[1]
                else:
                    auth_token = None
                    
                detector = AgeGenderDetector()
                if detector.start_detection(video_path, camera_id=camera_id, auth_token=auth_token):
                    age_gender_detector_instances[camera_id] = detector
                    return JsonResponse({
                        'status': 'success',
                        'message': 'Camera connected'
                    })
                else:
                    camera.delete()  # Clean up if detector fails to start
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Failed to start detection'
                    })
            
            return JsonResponse({
                'status': 'success',
                'message': 'Camera already connected'
            })
            
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })

@api_view(['GET'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def get_cameras(request):
    try:
        cameras = Camera.objects.filter(
            user=request.user,
            camera_type='shoplift',
            is_active=True
        )
        
        # Get auth token from request
        token = None
        if 'HTTP_AUTHORIZATION' in request.META and request.META['HTTP_AUTHORIZATION'].startswith('Bearer '):
            token = request.META['HTTP_AUTHORIZATION'].split(' ')[1]
        
        if not token:
            # Generate a new token if not available in header
            refresh = RefreshToken.for_user(request.user)
            token = str(refresh.access_token)
            
        print(f"Token available for camera activation: {bool(token)}")
        
        # Clean up any detectors for inactive cameras
        with detector_lock:
            all_camera_ids = set(cam.camera_id for cam in cameras)
            stale_detector_ids = set(detector_instances.keys()) - all_camera_ids
            
            for stale_id in stale_detector_ids:
                print(f"Cleaning up stale detector for camera {stale_id}")
                try:
                    detector = detector_instances[stale_id]
                    detector.is_stopping = True
                    detector.stop_detection()
                    del detector_instances[stale_id]
                except Exception as e:
                    print(f"Error cleaning up stale detector: {str(e)}")
        
        camera_list = []
        for camera in cameras:
            camera_id = camera.camera_id
            # Check if video file exists before starting detection
            if not os.path.exists(camera.video_path):
                print(f"Warning: Video file not found for camera {camera_id} at {camera.video_path}. Skipping this camera.")
                continue
            with detector_lock:
                if camera_id not in detector_instances:
                    print(f"Initializing detector for camera {camera_id}")
                    detector = ShopliftDetector()
                    try:
                        if detector.start_detection(camera.video_path, camera_id, token):
                            detector_instances[camera_id] = detector
                            print(f"Successfully started detection for camera {camera_id}")
                        else:
                            print(f"Failed to start detection for camera {camera_id}")
                    except FileNotFoundError as e:
                        print(f"Error: {e}. Skipping camera {camera_id}.")
                        continue
                else:
                    # Update the auth token in existing detector
                    detector = detector_instances[camera_id]
                    if detector.auth_token != token:
                        print(f"Updating auth token for camera {camera_id}")
                        detector.auth_token = token
            camera_list.append({
                'id': camera_id,
                'path': camera.video_path,
                'name': camera.name or f"Camera {camera.id}"
            })
        
        return JsonResponse({
            'status': 'success',
            'cameras': camera_list
        })
        
    except Exception as e:
        print(f"Error getting cameras: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@api_view(['GET'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def get_age_gender_cameras(request):
    try:
        cameras = Camera.objects.filter(
            user=request.user,
            camera_type='age_gender',
            is_active=True
        )
        
        camera_list = []
        for camera in cameras:
            camera_list.append({
                'camera_id': camera.camera_id,
                'name': camera.name,  # Add this
                'video_path': camera.video_path
            })
            
            # Start detection if not already running
            if camera.camera_id not in age_gender_detector_instances:
                detector = AgeGenderDetector()
                # Get auth token for API calls
                auth_header = request.headers.get('Authorization')
                if auth_header and auth_header.startswith('Bearer '):
                    auth_token = auth_header.split(' ')[1]
                else:
                    auth_token = None
                    
                if detector.start_detection(camera.video_path, camera_id=camera.camera_id, auth_token=auth_token):
                    age_gender_detector_instances[camera.camera_id] = detector

        return JsonResponse({
            'status': 'success',
            'cameras': camera_list
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_camera(request, camera_id):
    try:
        with detector_lock:
            camera = Camera.objects.filter(
                camera_id=camera_id,
                user=request.user
            ).first()
            
            if camera:
                # Stop and clean up detector instance
                if camera_id in detector_instances:
                    try:
                        detector = detector_instances[camera_id]
                        print(f"Stopping detection for camera {camera_id}")
                        
                        # Set a flag to prevent further alert generation
                        detector.is_stopping = True
                        
                        # Stop detection (should terminate threads)
                        detector.stop_detection()
                        
                        # Clear any alert-related state
                        if hasattr(detector, 'current_alert_id'):
                            print(f"Cleaning up in-progress alert for camera {camera_id}")
                            try:
                                from .models import ShopliftingAlert
                                # Get any in-progress alerts for this camera
                                in_progress_alerts = ShopliftingAlert.objects.filter(
                                    camera=camera,
                                    status='recording_in_progress'
                                )
                                # Delete these alerts to prevent notifications
                                if in_progress_alerts.exists():
                                    print(f"Deleting {in_progress_alerts.count()} in-progress alerts")
                                    in_progress_alerts.delete()
                            except Exception as alert_err:
                                print(f"Error cleaning up alerts: {alert_err}")
                        
                        # Allow time for threads to terminate gracefully
                        import time
                        time.sleep(0.5)
                        
                        # Remove from instances dictionary
                        del detector_instances[camera_id]
                        print(f"Detector instance for camera {camera_id} removed")
                    except Exception as detector_err:
                        print(f"Error stopping detector: {detector_err}")
                
                # Mark camera as inactive
                camera.is_active = False
                camera.save()
                
                # Force Python garbage collection to clean up resources
                import gc
                gc.collect()
                
                return JsonResponse({
                    'status': 'success',
                    'message': 'Camera deleted and all resources cleaned up'
                })
                
        return JsonResponse({
            'status': 'error',
            'message': 'Camera not found'
        })
    except Exception as e:
        print(f"Error deleting camera: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@api_view(['DELETE'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def delete_age_gender_camera(request, camera_id):
    try:
        with age_gender_detector_lock:
            camera = Camera.objects.filter(
                camera_id=camera_id,
                user=request.user,
                camera_type='age_gender'
            ).first()
            
            if camera:
                # Stop and cleanup detector
                if camera_id in age_gender_detector_instances:
                    try:
                        detector = age_gender_detector_instances[camera_id]
                        print(f"Stopping age-gender detection for camera {camera_id}")
                        detector.stop_detection()
                        
                        # Remove from instances dictionary
                        del age_gender_detector_instances[camera_id]
                        print(f"Age-gender detector instance for camera {camera_id} removed")
                    except Exception as detector_err:
                        print(f"Error stopping age-gender detector: {detector_err}")
                
                # Mark camera as inactive instead of deleting it
                camera.is_active = False
                camera.save()
                
                # Force Python garbage collection to clean up resources
                import gc
                gc.collect()
                
                return JsonResponse({
                    'status': 'success',
                    'message': 'Camera deleted successfully and all resources cleaned up'
                })
            
            return JsonResponse({
                'status': 'error',
                'message': 'Camera not found'
            }, status=404)
            
    except Exception as e:
        print(f"Error deleting camera: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@api_view(['GET'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def get_frame(request, camera_id):
    try:
        # Ensure the camera belongs to the user and is active
        try:
            camera = Camera.objects.get(
                camera_id=camera_id,
                user=request.user,
                is_active=True
            )
        except Camera.DoesNotExist:
            # If camera is not active or doesn't exist, check if we need to stop any detectors
            with detector_lock:
                if camera_id in detector_instances:
                    # Camera exists in detector instances but is no longer active - stop it
                    print(f"Camera {camera_id} is inactive or deleted but detector is running - stopping it")
                    try:
                        detector = detector_instances[camera_id]
                        detector.is_stopping = True
                        detector.stop_detection()
                        del detector_instances[camera_id]
                        # Force garbage collection
                        import gc
                        gc.collect()
                    except Exception as e:
                        print(f"Error stopping detector for inactive camera: {str(e)}")
            
            return JsonResponse({
                'status': 'error',
                'message': 'Camera not found or inactive'
            }, status=404)
        
        # Get or create detector instance
        with detector_lock:
            if camera_id not in detector_instances:
                detector = ShopliftDetector()
                # Get authentication token
                token = request.META.get('HTTP_AUTHORIZATION', '').split(' ')[1] if 'HTTP_AUTHORIZATION' in request.META else None
                if not token:
                    # If token not in header, try to generate a new one
                    refresh = RefreshToken.for_user(request.user)
                    token = str(refresh.access_token)
                    
                print(f"Starting detection for camera {camera_id} with token")
                if detector.start_detection(camera.video_path, camera_id, token):
                    detector_instances[camera_id] = detector
                else:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Failed to start detection'
                    }, status=500)
            else:
                detector = detector_instances[camera_id]
                # Update auth token if needed
                token = request.META.get('HTTP_AUTHORIZATION', '').split(' ')[1] if 'HTTP_AUTHORIZATION' in request.META else None
                if token and detector.auth_token != token:
                    print(f"Updating auth token for camera {camera_id}")
                    detector.auth_token = token

        # Get frame from detector with frame rate control
        frame_data = detector.get_frame()
        if frame_data:
            # Add frame timestamp to help with synchronization
            import time
            current_time = time.time()
            return JsonResponse({
                'status': 'success',
                'frame': frame_data,
                'timestamp': current_time,
                'frame_rate': 30  # Set a consistent frame rate
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'No frame available'
            }, status=404)
    except Exception as e:
        print(f"Error getting frame: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@api_view(['GET'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def get_age_gender_frame(request, camera_id):
    try:
        with age_gender_detector_lock:
            detector = age_gender_detector_instances.get(camera_id)
            if detector:
                frame_data = detector.get_frame()
                if frame_data:
                    return JsonResponse({
                        'status': 'success',
                        'frame': frame_data
                    })
    except Exception as e:
        pass
        
    return JsonResponse({
        'status': 'error',
        'message': 'No frame available'
    })

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def map_view(request):
    if request.method == 'GET':
        try:
            map_obj = Map.objects.filter(user=request.user).first()
            if map_obj:
                return Response({
                    'iframe_code': map_obj.iframe_code,
                    'editor_link': map_obj.editor_link
                })
            return Response({
                'iframe_code': '',
                'editor_link': ''
            })
        except Exception as e:
            return Response({
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    elif request.method == 'POST':
        try:
            iframe_code = request.data.get('iframe_code')
            editor_link = request.data.get('editor_link')
            if not iframe_code:
                return Response({
                    'message': 'iframe_code is required'
                }, status=status.HTTP_400_BAD_REQUEST)

            map_obj, created = Map.objects.get_or_create(
                user=request.user,
                defaults={
                    'iframe_code': iframe_code,
                    'editor_link': editor_link
                }
            )
            
            if not created:
                map_obj.iframe_code = iframe_code
                map_obj.editor_link = editor_link
                map_obj.save()

            return Response({
                'message': 'Map saved successfully'
            })
        except Exception as e:
            return Response({
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Create your views here.

@api_view(['POST'])
def login_view(request):
    email = request.data.get('email')
    password = request.data.get('password')
    
    try:
        # Get user by email 
        user = User.objects.get(email=email)
        
        # Try direct password verification
        if user.check_password(password): 
            # Check if email is verified
            try:
                profile = UserProfile.objects.get(user=user)
                if not profile.email_verified:
                    return Response({
                        'status': 'error',
                        'message': 'Please verify your email before logging in. Check your inbox for the verification link.'
                    }, status=status.HTTP_401_UNAUTHORIZED)
                
                profile_data = {
                    'title': profile.title,
                    'company': profile.company,
                    'phone': profile.phone
                }
            except UserProfile.DoesNotExist:
                profile_data = {
                    'title': '',
                    'company': '',
                    'phone': ''
                }
            
            refresh = RefreshToken.for_user(user)
            return Response({
                'status': 'success',
                'message': 'Login successful',
                'tokens': {
                    'access': str(refresh.access_token),
                    'refresh': str(refresh),
                },
                'user': {
                    'email': user.email,
                    'id': user.id,
                    'username': user.username,
                    'first_name': user.first_name or '',
                    'last_name': user.last_name or '',
                    'is_staff': user.is_staff,
                    'is_superuser': user.is_superuser,
                    **profile_data
                }
            })
        else:
            return Response({
                'status': 'error',
                'message': 'Invalid password'
            }, status=status.HTTP_401_UNAUTHORIZED)
            
    except User.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'No account found with this email'
        }, status=status.HTTP_401_UNAUTHORIZED)
    except Exception as e:
        print(f"Login error: {str(e)}")
        return Response({
            'status': 'error',
            'message': 'An error occurred during login'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def reset_password(request):
    try:
        email = request.data.get('email')
        user = User.objects.get(email=email)
        
        # Generate token
        token = default_token_generator.make_token(user)
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        
        # Send reset email
        subject = 'Password Reset Request'
        message = f'''
        Click the following link to reset your password:
        {settings.FRONTEND_URL}/reset-password?uid={uid}&token={token}
        
        This link will expire in 24 hours.
        '''
        
        send_mail(
            subject=subject,
            message=message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[email],
            fail_silently=False,
        )
        
        return Response({'status': 'success', 'message': 'Password reset instructions sent'})
    except User.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'No account found with this email'
        }, status=status.HTTP_404_NOT_FOUND)


@api_view(['POST'])
def contact_form(request):
    try:
        # Extract form data
        first_name = request.data.get('firstName')
        last_name = request.data.get('lastName')
        email = request.data.get('email')
        phone = request.data.get('phone')
        message = request.data.get('message')

        # Compose email
        subject = f'New Contact Form Submission from {first_name} {last_name}'
        email_body = f"""
        New contact form submission received:

        Name: {first_name} {last_name}
        Email: {email}
        Phone: {phone or 'Not provided'}

        Message:
        {message}
        """

        try:
            send_mail(
                subject=subject,
                message=email_body,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[settings.DEFAULT_FROM_EMAIL],  # Send to the same Gmail address
                fail_silently=False,
            )
            
            return Response({
                'status': 'success',
                'message': 'Thank you for your message. We will get back to you soon!'
            })
            
        except Exception as email_error:
            print(f"Email error: {str(email_error)}")
            return Response({
                'status': 'error',
                'message': 'Failed to send email. Please try again.'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except Exception as e:
        print(f"General error: {str(e)}")
        return Response({
            'status': 'error',
            'message': 'An error occurred. Please try again.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def update_profile(request):
    if not request.user.is_authenticated:
        return Response({'status': 'error', 'message': 'Not authenticated'}, status=401)

    try:
        user = request.user
        data = request.data

        # Update basic user fields
        if 'first_name' in data:
            user.first_name = data['first_name']
        if 'last_name' in data:
            user.last_name = data['last_name']
        if 'email' in data:
            # Check if email is unique
            if User.objects.filter(email=data['email']).exclude(id=user.id).exists():
                return Response({
                    'status': 'error',
                    'message': 'Email already exists'
                }, status=400)
            user.email = data['email']
        
        # Update password if provided
        if 'password' in data and data['password']:
            user.set_password(data['password'])

        # Update profile fields
        profile = user.userprofile
        if 'title' in data:
            profile.title = data['title']
        if 'company' in data:
            profile.company = data['company']

        user.save()
        profile.save()

        # Return updated user data
        return Response({
            'status': 'success',
            'message': 'Profile updated successfully',
            'user': {
                'id': user.id,
                'email': user.email,
                'username': user.username,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'title': profile.title,
                'company': profile.company,
            }
        })

    except Exception as e:
        return Response({
            'status': 'error',
            'message': str(e)
        }, status=400)


@csrf_exempt
def signup(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed'}, status=405)

    try:
        data = json.loads(request.body)
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        first_name = data.get('firstName', '')
        last_name = data.get('lastName', '')
        title = data.get('title', '')
        company = data.get('company', '')

        # Validate required fields
        if not all([email, password, first_name, last_name, title, company]):
            return JsonResponse({
                'status': 'error',
                'message': 'All fields are required'
            }, status=400)

        # Check if user already exists
        if User.objects.filter(email=email).exists():
            return JsonResponse({
                'status': 'error',
                'message': 'A user with this email already exists'
            }, status=400)

        # Create user
        user = User.objects.create_user(
            username=email,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
            is_active=True
        )

        # Generate verification token
        verification_token = str(uuid.uuid4())
        token_expires = timezone.now() + timedelta(hours=24)

        # Create user profile
        UserProfile.objects.create(
            user=user,
            title=title,
            company=company,
            verification_token=verification_token,
            verification_token_expires=token_expires
        )

        # Send verification email
        frontend_url = settings.FRONTEND_URL
        verification_link = f"{frontend_url}/verify-email?token={verification_token}"
        send_mail(
            'Verify your email - Retail Vista',
            f'''Welcome to Retail Vista!

Please click the following link to verify your email:
{verification_link}

This link will expire in 24 hours.''',
            settings.DEFAULT_FROM_EMAIL,
            [email],
            fail_silently=False,
        )

        return JsonResponse({
            'status': 'success',
            'message': 'Please check your email to verify your account.'
        })

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@csrf_exempt
def verify_email(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed'}, status=405)

    try:
        data = json.loads(request.body)
        token = data.get('token', '')

        if not token:
            return JsonResponse({
                'status': 'error',
                'message': 'Verification token is required'
            }, status=400)

        # Find user profile with this token
        try:
            profile = UserProfile.objects.get(
                verification_token=token,
                verification_token_expires__gt=timezone.now(),
                email_verified=False
            )
        except UserProfile.DoesNotExist:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid or expired verification token'
            }, status=400)

        # Mark email as verified
        profile.email_verified = True
        profile.verification_token = ''
        profile.save()

        return JsonResponse({
            'status': 'success',
            'message': 'Email verified successfully'
        })

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@api_view(['GET'])
@permission_classes([IsAdminUser])
def get_all_users(request):
    try:
        # Fetch all UserProfiles with related User data
        profiles = UserProfile.objects.select_related('user').all()
        users_data = []
        
        for profile in profiles:
            user = profile.user
            user_info = {
                'id': user.id,
                'store_id': profile.store_id,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'email': user.email,
                'title': profile.title,
                'company': profile.company,
                'email_verified': profile.email_verified,
                'created_at': profile.created_at,
                'updated_at': profile.updated_at,
                'role': 'Admin' if user.is_staff else 'User'
            }
            users_data.append(user_info)
        
        return Response({
            'status': 'success',
            'users': users_data
        }, status=status.HTTP_200_OK)
    
    except Exception as e:
        print(f"Error fetching users: {str(e)}")
        return Response({
            'status': 'error',
            'message': 'An error occurred while fetching users'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['DELETE'])
@permission_classes([IsAdminUser])
def delete_user(request, user_id):
    try:
        # Find the user to delete
        user = User.objects.get(id=user_id)
        
        # Prevent deleting the admin user
        if user.is_staff or user.is_superuser:
            return Response({
                'status': 'error',
                'message': 'Cannot delete admin or superuser accounts'
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Delete associated UserProfile if it exists
        try:
            profile = UserProfile.objects.get(user=user)
            profile.delete()
        except UserProfile.DoesNotExist:
            pass
        
        # Delete the user
        user.delete()
        
        return Response({
            'status': 'success',
            'message': 'User deleted successfully'
        }, status=status.HTTP_200_OK)
    
    except User.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'User not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        print(f"Error deleting user: {str(e)}")
        return Response({
            'status': 'error',
            'message': 'An error occurred while deleting the user'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAdminUser])
def admin_update_user(request):
    try:
        user_id = request.data.get('user_id')
        user = User.objects.get(id=user_id)
        
        # Update user fields
        user.first_name = request.data.get('first_name', user.first_name)
        user.last_name = request.data.get('last_name', user.last_name)
        user.email = request.data.get('email', user.email)
        user.save()

        # Update or create user profile
        profile, created = UserProfile.objects.get_or_create(user=user)
        profile.title = request.data.get('title', profile.title)
        profile.company = request.data.get('company', profile.company)
        profile.phone = request.data.get('phone', profile.phone)
        profile.email_verified = request.data.get('email_verified', profile.email_verified)
        profile.save()

        return Response({
            'status': 'success',
            'message': 'User updated successfully',
            'user': {
                'id': user.id,
                'store_id': profile.store_id,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'email': user.email,
                'title': profile.title,
                'company': profile.company,
                'phone': profile.phone,
                'email_verified': profile.email_verified,
                'created_at': profile.created_at,
                'updated_at': profile.updated_at
            }
        })
    except User.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'User not found'
        }, status=404)
    except Exception as e:
        return Response({
            'status': 'error',
            'message': str(e)
        }, status=400)

@api_view(['POST'])
def request_password_reset(request):
    try:
        email = request.data.get('email')
        user = User.objects.get(email=email)
        
        # Generate token and UID
        token = default_token_generator.make_token(user)
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        
        # Create reset link
        reset_link = f'{settings.FRONTEND_URL}/reset-password?uid={uid}&token={token}'
        
        # Send email
        subject = 'Password Reset Request'
        message = f'''
        Hello,
        
        You have requested to reset your password. Click the following link to reset your password:
        
        {reset_link}
        
        If you didn't request this, please ignore this email.
        
        This link will expire in 24 hours.
        
        Best regards,
        Retail Vista Team
        '''
        
        send_mail(
            subject=subject,
            message=message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[email],
            fail_silently=False,
        )
        
        return Response({
            'status': 'success',
            'message': 'Password reset instructions sent to your email'
        })
        
    except User.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'No account found with this email'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Password reset error: {str(e)}")
        return Response({
            'status': 'error',
            'message': 'An error occurred while processing your request'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def verify_reset_token(request):
    try:
        uid = request.data.get('uid')
        token = request.data.get('token')
        
        if not uid or not token:
            return Response({
                'status': 'error',
                'message': 'Invalid reset link'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        user_id = force_str(urlsafe_base64_decode(uid))
        user = User.objects.get(pk=user_id)
        
        if default_token_generator.check_token(user, token):
            return Response({
                'status': 'success',
                'message': 'Token is valid'
            })
        else:
            return Response({
                'status': 'error',
                'message': 'Invalid or expired reset link'
            }, status=status.HTTP_400_BAD_REQUEST)
            
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        return Response({
            'status': 'error',
            'message': 'Invalid reset link'
        }, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def set_new_password(request):
    try:
        uid = request.data.get('uid')
        token = request.data.get('token')
        new_password = request.data.get('password')
        
        if not all([uid, token, new_password]):
            return Response({
                'status': 'error',
                'message': 'Missing required fields'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Validate password
        is_valid, message = validate_password(new_password)
        if not is_valid:
            return Response({
                'status': 'error',
                'message': message
            }, status=status.HTTP_400_BAD_REQUEST)
        
        user_id = force_str(urlsafe_base64_decode(uid))
        user = User.objects.get(pk=user_id)
        
        if default_token_generator.check_token(user, token):
            user.set_password(new_password)
            user.save()
            return Response({
                'status': 'success',
                'message': 'Password has been reset successfully'
            })
        else:
            return Response({
                'status': 'error',
                'message': 'Invalid or expired reset link'
            }, status=status.HTTP_400_BAD_REQUEST)
            
    except Exception as e:
        logger.error(f"Password reset error: {str(e)}")
        return Response({
            'status': 'error',
            'message': 'An error occurred while resetting your password'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def promotion_list(request):
    if request.method == 'GET':
        # Get promotions for the current user only
        promotions = Promotion.objects.filter(user=request.user)
        serializer = PromotionSerializer(promotions, many=True, context={'request': request})
        return Response({
            'status': 'success',
            'promotions': serializer.data
        })

    elif request.method == 'POST':
        try:
            # Add user to the request data
            data = request.data.copy()
            serializer = PromotionSerializer(data=data, context={'request': request})
            
            if serializer.is_valid():
                # Save with the current user
                serializer.save(user=request.user)
                return Response({
                    'status': 'success',
                    'promotion': serializer.data
                }, status=status.HTTP_201_CREATED)
            
            return Response({
                'status': 'error',
                'errors': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
            
        except Exception as e:
            return Response({
                'status': 'error',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def active_promotions(request):
    # Get only active promotions for the current user
    promotions = Promotion.objects.filter(
        user=request.user,
        status='active'
    )
    serializer = PromotionSerializer(promotions, many=True, context={'request': request})
    return Response({
        'status': 'success',
        'promotions': serializer.data
    })

@api_view(['GET', 'PUT', 'DELETE'])
@permission_classes([IsAuthenticated])
def promotion_detail(request, pk):
    try:
        promotion = Promotion.objects.get(pk=pk, user=request.user)
    except Promotion.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Promotion not found'
        }, status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = PromotionSerializer(promotion, context={'request': request})
        return Response({
            'status': 'success',
            'promotion': serializer.data
        })

    elif request.method == 'PUT':
        # Create a mutable copy of the data
        data = request.data.copy()
        
        # If no new image is provided, keep the existing one
        if 'image' not in request.FILES:
            data.pop('image', None)
        
        serializer = PromotionSerializer(
            promotion,
            data=data,
            partial=True,
            context={'request': request}
        )
        
        if serializer.is_valid():
            serializer.save()
            return Response({
                'status': 'success',
                'promotion': serializer.data
            })
        return Response({
            'status': 'error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        promotion.delete()
        return Response({
            'status': 'success',
            'message': 'Promotion deleted successfully'
        }, status=status.HTTP_204_NO_CONTENT)

@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def connect_people_counter(request):
    camera_id = request.data.get('camera_id')
    video_path = request.data.get('video_path')
    name = request.data.get('name')  # Add this
    entry_points = request.data.get('entry_points')
    exit_points = request.data.get('exit_points')
    
    try:
        camera = Camera.objects.create(
            user=request.user,
            camera_id=camera_id,
            name=name,  # Add this
            video_path=video_path,
            camera_type='people_counter',
            is_active=True
        )
        
        detector = PeopleCounterDetector(camera_id)
        detector.start_detection(video_path, entry_points=entry_points, exit_points=exit_points)
        people_counter_detectors[camera_id] = detector
        
        return Response({'status': 'success'})
    except Exception as e:
        return Response({'status': 'error', 'message': str(e)})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_people_counter_cameras(request):
    cameras = Camera.objects.filter(
        user=request.user,
        camera_type='people_counter',
        is_active=True
    )
    return Response({
        'status': 'success',
        'cameras': [{
            'id': cam.camera_id,
            'name': cam.name,  # Add this line
            'count': cam.entry_count
        } for cam in cameras]
    })

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_people_counter_camera(request, camera_id):
    try:
        camera = Camera.objects.filter(
            user=request.user,
            camera_id=camera_id,
            camera_type='people_counter'
        ).first()
        
        if not camera:
            return Response({
                'status': 'error',
                'message': 'Camera not found'
            }, status=404)
        
        # Stop and cleanup detector
        if camera_id in people_counter_detectors:
            try:
                detector = people_counter_detectors[camera_id]
                print(f"Stopping people counter detection for camera {camera_id}")
                detector.stop_detection()
                
                # Remove from instances dictionary
                del people_counter_detectors[camera_id]
                print(f"People counter detector instance for camera {camera_id} removed")
            except Exception as detector_err:
                print(f"Error stopping people counter detector: {detector_err}")
        
        # Mark camera as inactive
        camera.is_active = False
        camera.save()
        
        # Force Python garbage collection to clean up resources
        import gc
        gc.collect()
        
        return Response({
            'status': 'success',
            'message': 'Camera deleted successfully and all resources cleaned up'
        })
    except Exception as e:
        print(f"Error deleting people counter camera: {str(e)}")
        import traceback
        traceback.print_exc()
        return Response({
            'status': 'error',
            'message': str(e)
        }, status=500)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_people_counter_frame(request, camera_id):
    try:
        detector = people_counter_detectors.get(camera_id)
        if detector:
            frame = detector.get_frame()
            count = detector.get_count()
            
            # Convert entry/exit points to lists if they're numpy arrays
            entry_points = [np.array(points).tolist() if isinstance(points, np.ndarray) else points 
                          for points in detector.entry_points]
            exit_points = [np.array(points).tolist() if isinstance(points, np.ndarray) else points 
                          for points in detector.exit_points]
            
            return Response({
                'status': 'success',
                'frame': frame,
                'count': count,
                'entry_points': entry_points,
                'exit_points': exit_points
            })
        return Response({'status': 'error', 'message': 'Camera not found'})
    except Exception as e:
        return Response({'status': 'error', 'message': str(e)})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_people_counter_data(request):
    try:
        from .models import PeopleCounterData
        camera_id = request.data.get('camera_id')
        entry_count = request.data.get('entry_count', 0)
        zone_name = request.data.get('zone_name', 'Main Zone')
        
        # Get the camera
        try:
            camera = Camera.objects.get(
                user=request.user, 
                camera_id=camera_id,
                camera_type='people_counter'
            )
            
            # Create a new data entry
            PeopleCounterData.objects.create(
                camera=camera,
                entry_count=entry_count,
                zone_name=zone_name
            )
            
            return Response({'status': 'success'})
        except Camera.DoesNotExist:
            return Response({'status': 'error', 'message': 'Camera not found'}, status=404)
            
    except Exception as e:
        return Response({'status': 'error', 'message': str(e)}, status=500)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_analysis_data(request):
    time_filter = request.GET.get('time_filter', '1h')  # Default to 1 hour
    
    # Calculate time range based on filter
    now = timezone.now()
    if time_filter == '1h':
        start_time = now - timedelta(hours=1)
        trunc_func = TruncHour
        interval = timedelta(minutes=5)  # Changed from 10 seconds to 5 minutes
    elif time_filter == '7d':
        start_time = now - timedelta(days=7)
        trunc_func = TruncDay
        interval = timedelta(hours=6)  # Changed from 30 minutes to 6 hours
    elif time_filter == '30d':
        start_time = now - timedelta(days=30)
        trunc_func = TruncDay
        interval = timedelta(days=1)  # Changed from 2 hours to 1 day
    elif time_filter == '1y':
        start_time = now - timedelta(days=365)
        trunc_func = TruncMonth
        interval = timedelta(days=7)  # Changed from 1 day to 1 week
    else:
        return Response({'error': 'Invalid time filter'}, status=400)

    # Get real-time stats for last 10 seconds only
    last_10_seconds = now - timedelta(seconds=10)
    recent_data = DetectionData.objects.filter(
        camera__user=request.user,
        timestamp__gte=last_10_seconds
    )
    
    total_suspicious_frames = recent_data.aggregate(
        total=Sum('suspicious_events')
    )['total'] or 0
    
    # Calculate current average suspicious rate from all data points in the selected time range
    all_cameras_data = DetectionData.objects.filter(
        camera__user=request.user,
        timestamp__gte=start_time
    ).aggregate(
        total_events=Sum('total_events'),
        suspicious_events=Sum('suspicious_events')
    )
    
    total_events = all_cameras_data['total_events'] or 0
    suspicious_events = all_cameras_data['suspicious_events'] or 0
    current_avg_rate = (suspicious_events / total_events * 100) if total_events > 0 else 0
    
    # Get active cameras count
    active_cameras = Camera.objects.filter(
        user=request.user,
        camera_type='shoplift',
        is_active=True
    ).count()
    
    # Generate time points for the selected range
    time_points = []
    current_time = start_time
    while current_time <= now:
        time_points.append(current_time)
        current_time += interval

    # Get time series data for line graph with proper intervals
    time_series = []
    for point in time_points:
        point_data = DetectionData.objects.filter(
            camera__user=request.user,
            timestamp__lte=point,
            timestamp__gt=point - interval
        ).aggregate(
            total_events=Sum('total_events'),
            suspicious_events=Sum('suspicious_events')
        )
        
        total = point_data['total_events'] or 0
        suspicious = point_data['suspicious_events'] or 0
        percentage = (suspicious / total * 100) if total > 0 else 0
        
        time_series.append({
            'timestamp': point,
            'percentage': round(percentage, 2)
        })
    
    # Get camera comparison data for bar graph
    camera_data = DetectionData.objects.filter(
        camera__user=request.user,
        timestamp__gte=start_time
    ).values('camera__name').annotate(
        suspicious_events=Sum('suspicious_events')
    )
    
    total_suspicious = sum(entry['suspicious_events'] for entry in camera_data)
    camera_comparison = [{
        'camera_name': entry['camera__name'] or f"Camera {idx + 1}",
        'percentage': round((entry['suspicious_events'] / total_suspicious * 100), 2)
        if total_suspicious > 0 else 0
    } for idx, entry in enumerate(camera_data)]
    
    return Response({
        'status': 'success',
        'data': {
            'realtime_stats': {
                'total_suspicious_frames': total_suspicious_frames,
                'current_avg_rate': round(current_avg_rate, 2),
                'active_cameras': active_cameras
            },
            'time_series': time_series,
            'camera_comparison': camera_comparison
        }
    })

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_people_counter_analysis(request):
    from .models import PeopleCounterData
    time_filter = request.GET.get('time_filter', '1h')  # Default to 1 hour
    
    # Calculate time range based on filter
    now = timezone.now()
    if time_filter == '1h':
        start_time = now - timedelta(hours=1)
        interval = timedelta(minutes=5)
    elif time_filter == '7d':
        start_time = now - timedelta(days=7)
        interval = timedelta(hours=6)
    elif time_filter == '30d':
        start_time = now - timedelta(days=30)
        interval = timedelta(days=1)
    elif time_filter == '1y':
        start_time = now - timedelta(days=365)
        interval = timedelta(days=7)
    else:
        return Response({'error': 'Invalid time filter'}, status=400)

    # Get recent data for real-time stats (last 10 seconds)
    last_10_seconds = now - timedelta(seconds=10)
    
    # Get active cameras count
    active_cameras = Camera.objects.filter(
        user=request.user,
        camera_type='people_counter',
        is_active=True
    ).count()
    
    # Get total entry count (from all cameras)
    total_entries = Camera.objects.filter(
        user=request.user,
        camera_type='people_counter'
    ).aggregate(total=Sum('entry_count'))['total'] or 0
    
    # Get most active zone
    zone_data = PeopleCounterData.objects.filter(
        camera__user=request.user,
        camera__camera_type='people_counter',
        timestamp__gte=start_time
    ).values('zone_name').annotate(
        total_entries=Sum('entry_count')
    ).order_by('-total_entries')
    
    most_active_zone = "N/A"
    most_active_zone_count = 0
    
    if zone_data.exists():
        most_active_zone = zone_data[0]['zone_name']
        most_active_zone_count = zone_data[0]['total_entries']
    
    # Get least active zone
    least_active_zone = "N/A"
    least_active_zone_count = 0
    
    if zone_data.exists() and len(zone_data) > 1:
        least_active_zone = zone_data.last()['zone_name']
        least_active_zone_count = zone_data.last()['total_entries']
    
    # Generate time points for the selected range
    time_points = []
    current_time = start_time
    while current_time <= now:
        time_points.append(current_time)
        current_time += interval
    
    # Get time series data for zone traffic over time
    time_labels = []
    zone_entries = defaultdict(list)
    
    for point in time_points:
        # Format the label based on the time filter
        if time_filter == '1h':
            time_labels.append(point.strftime('%H:%M'))
        elif time_filter == '7d':
            time_labels.append(point.strftime('%a'))
        elif time_filter == '30d':
            time_labels.append(point.strftime('%d %b'))
        else:  # 1y
            time_labels.append(point.strftime('%b'))
            
        # Get data for each zone
        zone_results = PeopleCounterData.objects.filter(
            camera__user=request.user,
            camera__camera_type='people_counter',
            timestamp__lte=point,
            timestamp__gt=point - interval
        ).values('zone_name').annotate(
            total_entries=Sum('entry_count')
        )
        
        # Add data for each zone
        for zone in zone_data:
            zone_name = zone['zone_name']
            entries = 0
            
            # Find this zone in the results
            for result in zone_results:
                if result['zone_name'] == zone_name:
                    entries = result['total_entries']
                    break
                    
            zone_entries[zone_name].append(entries)
    
    # Get zone comparison data for bar graph
    zone_comparison = []
    total_zone_entries = sum(zone['total_entries'] for zone in zone_data)
    
    for zone in zone_data:
        percentage = 0
        if total_zone_entries > 0:
            percentage = round((zone['total_entries'] / total_zone_entries * 100), 2)
            
        zone_comparison.append({
            'zone_name': zone['zone_name'],
            'entries': zone['total_entries'],
            'percentage': percentage
        })
    
    # Prepare data for charts
    zone_traffic_data = {
        'labels': time_labels,
        'datasets': []
    }
    
    # Colors for different zones
    colors = [
        {'border': 'rgb(147, 51, 234)', 'bg': 'rgba(147, 51, 234, 0.5)'},  # Purple
        {'border': 'rgb(37, 99, 235)', 'bg': 'rgba(37, 99, 235, 0.5)'},     # Blue
        {'border': 'rgb(236, 72, 153)', 'bg': 'rgba(236, 72, 153, 0.5)'},   # Pink
        {'border': 'rgb(34, 197, 94)', 'bg': 'rgba(34, 197, 94, 0.5)'},     # Green
        {'border': 'rgb(234, 179, 8)', 'bg': 'rgba(234, 179, 8, 0.5)'}      # Yellow
    ]
    
    # Add datasets for each zone
    for idx, zone_name in enumerate(zone_entries.keys()):
        color_idx = idx % len(colors)
        zone_traffic_data['datasets'].append({
            'label': zone_name,
            'data': zone_entries[zone_name],
            'borderColor': colors[color_idx]['border'],
            'backgroundColor': colors[color_idx]['bg']
        })
    
    # Prepare zone comparison data for bar chart
    zone_comparison_data = {
        'labels': [zone['zone_name'] for zone in zone_comparison],
        'datasets': [{
            'label': 'Zone Traffic Distribution',
            'data': [zone['entries'] for zone in zone_comparison],
            'backgroundColor': [colors[idx % len(colors)]['bg'] for idx in range(len(zone_comparison))]
        }]
    }
    
    return Response({
        'status': 'success',
        'data': {
            'realtime_stats': {
                'most_active_zone': most_active_zone,
                'most_active_zone_count': most_active_zone_count,
                'total_entries': total_entries,
                'least_active_zone': least_active_zone,
                'least_active_zone_count': least_active_zone_count,
                'active_cameras': active_cameras
            },
            'zone_traffic_data': zone_traffic_data,
            'zone_comparison_data': zone_comparison_data
        }
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_detection_data(request):
    camera_id = request.data.get('camera_id')
    total_events = request.data.get('total_events', 0)
    suspicious_events = request.data.get('suspicious_events', 0)
    
    try:
        camera = Camera.objects.get(
            camera_id=camera_id,
            user=request.user,
            camera_type='shoplift'
        )
        
        DetectionData.objects.create(
            camera=camera,
            total_events=total_events,
            suspicious_events=suspicious_events
        )
        
        return Response({'status': 'success'})
    except Camera.DoesNotExist:
        return Response({'error': 'Camera not found'}, status=404)
    except Exception as e:
        return Response({'error': str(e)}, status=400)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_age_gender_analysis(request):
    from .models import AgeGenderDetectionData
    time_filter = request.GET.get('time_filter', '1h')  # Default to 1 hour
    
    # Calculate time range based on filter
    now = timezone.now()
    if time_filter == '1h':
        start_time = now - timedelta(hours=1)
        interval = timedelta(minutes=5)
    elif time_filter == '7d':
        start_time = now - timedelta(days=7)
        interval = timedelta(hours=6)
    elif time_filter == '30d':
        start_time = now - timedelta(days=30)
        interval = timedelta(days=1)
    elif time_filter == '1y':
        start_time = now - timedelta(days=365)
        interval = timedelta(days=7)
    else:
        return Response({'error': 'Invalid time filter'}, status=400)

    # Get recent data for real-time stats (last 10 seconds)
    last_10_seconds = now - timedelta(seconds=10)
    recent_data = AgeGenderDetectionData.objects.filter(
        camera__user=request.user,
        camera__camera_type='age_gender',
        timestamp__gte=last_10_seconds
    )
    
    # Calculate most common age group
    recent_age_data = recent_data.aggregate(
        age_0_3=Sum('age_0_3_count'),
        age_4_7=Sum('age_4_7_count'),
        age_8_12=Sum('age_8_12_count'),
        age_13_20=Sum('age_13_20_count'),
        age_21_32=Sum('age_21_32_count'),
        age_33_43=Sum('age_33_43_count'),
        age_44_53=Sum('age_44_53_count'),
        age_60_100=Sum('age_60_100_count')
    )
    
    # Initialize age group if no data
    if not recent_age_data['age_0_3']:
        most_common_age = 'N/A'  # Changed from '21-32' to 'N/A' when no data is available
    else:
        # Find the age group with the highest count
        age_counts = {
            '0-3': recent_age_data['age_0_3'] or 0,
            '4-7': recent_age_data['age_4_7'] or 0,
            '8-12': recent_age_data['age_8_12'] or 0,
            '13-20': recent_age_data['age_13_20'] or 0,
            '21-32': recent_age_data['age_21_32'] or 0,
            '33-43': recent_age_data['age_33_43'] or 0,
            '44-53': recent_age_data['age_44_53'] or 0,
            '60-100': recent_age_data['age_60_100'] or 0,
        }
        most_common_age = max(age_counts.items(), key=lambda x: x[1])[0]
    
    # Calculate gender ratio
    gender_data = recent_data.aggregate(
        male=Sum('male_count'),
        female=Sum('female_count')
    )
    
    male_count = gender_data['male'] or 0
    female_count = gender_data['female'] or 0
    
    # Format ratio to one decimal place or handle division by zero
    if male_count == 0 and female_count == 0:
        gender_ratio = "N/A"
    elif female_count == 0:
        gender_ratio = f"{male_count}:0"
    else:
        ratio = round(male_count / female_count, 1)
        gender_ratio = f"{ratio}:1"
    
    # Get active cameras count
    active_cameras = Camera.objects.filter(
        user=request.user,
        camera_type='age_gender',
        is_active=True
    ).count()
    
    # Generate time points for the selected range
    time_points = []
    current_time = start_time
    while current_time <= now:
        time_points.append(current_time)
        current_time += interval
    
    # Get time series data for gender distribution over time
    time_series_male = []
    time_series_female = []
    time_labels = []
    
    for point in time_points:
        point_data = AgeGenderDetectionData.objects.filter(
            camera__user=request.user,
            camera__camera_type='age_gender',
            timestamp__lte=point,
            timestamp__gt=point - interval
        ).aggregate(
            male=Sum('male_count'),
            female=Sum('female_count')
        )
        
        # Format time label based on time filter
        if time_filter == '1h':
            time_label = point.strftime('%H:%M')
        elif time_filter == '7d':
            time_label = point.strftime('%a')
        elif time_filter == '30d':
            time_label = point.strftime('%b %d')
        else:
            time_label = point.strftime('%b')
            
        time_labels.append(time_label)
        time_series_male.append(point_data['male'] or 0)
        time_series_female.append(point_data['female'] or 0)
    
    # Get age distribution data
    age_distribution = AgeGenderDetectionData.objects.filter(
        camera__user=request.user,
        camera__camera_type='age_gender',
        timestamp__gte=start_time
    ).aggregate(
        age_0_3=Sum('age_0_3_count'),
        age_4_7=Sum('age_4_7_count'),
        age_8_12=Sum('age_8_12_count'),
        age_13_20=Sum('age_13_20_count'),
        age_21_32=Sum('age_21_32_count'),
        age_33_43=Sum('age_33_43_count'),
        age_44_53=Sum('age_44_53_count'),
        age_60_100=Sum('age_60_100_count')
    )

    age_distribution_data = [
        age_distribution['age_0_3'] or 0,
        age_distribution['age_4_7'] or 0,
        age_distribution['age_8_12'] or 0,
        age_distribution['age_13_20'] or 0,
        age_distribution['age_21_32'] or 0,
        age_distribution['age_33_43'] or 0,
        age_distribution['age_44_53'] or 0,
        age_distribution['age_60_100'] or 0
    ]
    
    # Get gender distribution data
    gender_distribution = AgeGenderDetectionData.objects.filter(
        camera__user=request.user,
        camera__camera_type='age_gender',
        timestamp__gte=start_time
    ).aggregate(
        male=Sum('male_count'),
        female=Sum('female_count')
    )
    
    gender_distribution_data = [
        gender_distribution['male'] or 0,
        gender_distribution['female'] or 0
    ]

    return Response({
        'status': 'success',
        'data': {
            'realtime_stats': {
                'most_common_age': most_common_age,
                'gender_ratio': gender_ratio,
                'active_cameras': active_cameras
            },
            'time_series': {
                'labels': time_labels,
                'male': time_series_male,
                'female': time_series_female
            },
            'age_distribution': age_distribution_data,
            'gender_distribution': gender_distribution_data
        }
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_age_gender_data(request):
    from .models import AgeGenderDetectionData
    
    camera_id = request.data.get('camera_id')
    male_count = request.data.get('male_count', 0)
    female_count = request.data.get('female_count', 0)
    
    # Age group counts
    age_0_3_count = request.data.get('age_0_3_count', 0)
    age_4_7_count = request.data.get('age_4_7_count', 0)
    age_8_12_count = request.data.get('age_8_12_count', 0)
    age_13_20_count = request.data.get('age_13_20_count', 0)
    age_21_32_count = request.data.get('age_21_32_count', 0)
    age_33_43_count = request.data.get('age_33_43_count', 0)
    age_44_53_count = request.data.get('age_44_53_count', 0)
    age_60_100_count = request.data.get('age_60_100_count', 0)
    
    total_detections = male_count + female_count
    
    try:
        camera = Camera.objects.get(
            camera_id=camera_id,
            user=request.user,
            camera_type='age_gender'
        )
        
        AgeGenderDetectionData.objects.create(
            camera=camera,
            male_count=male_count,
            female_count=female_count,
            age_0_3_count=age_0_3_count,
            age_4_7_count=age_4_7_count,
            age_8_12_count=age_8_12_count,
            age_13_20_count=age_13_20_count,
            age_21_32_count=age_21_32_count,
            age_33_43_count=age_33_43_count,
            age_44_53_count=age_44_53_count,
            age_60_100_count=age_60_100_count,
            total_detections=total_detections
        )
        
        return Response({'status': 'success'})
    except Camera.DoesNotExist:
        return Response({'error': 'Camera not found'}, status=404)
    except Exception as e:
        return Response({'error': str(e)}, status=400)

@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def get_first_frame(request):
    video_path = request.data.get('video_path')
    if not video_path:
        return Response({'status': 'error', 'message': 'No video path provided'})
        
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return Response({'status': 'error', 'message': 'Could not open video'})
            
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return Response({'status': 'error', 'message': 'Could not read frame'})
            
        frame = cv2.resize(frame, (1020, 600))
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return Response({
            'status': 'success',
            'frame': frame_base64
        })
    except Exception as e:
        return Response({'status': 'error', 'message': str(e)})

@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def create_shoplifting_alert(request):
    try:
        print("Received shoplifting alert request")
        camera_id = request.data.get('camera_id')
        if not camera_id:
            return Response({'error': 'Camera ID is required'}, status=400)
            
        # Get camera
        try:
            camera = Camera.objects.get(
                camera_id=camera_id,
                user=request.user,
                camera_type='shoplift',
                is_active=True  # Only allow alerts for active cameras
            )
        except Camera.DoesNotExist:
            return Response({'error': f'Camera not found or inactive with ID {camera_id}'}, status=404)
            
        # Check if files are in the request
        video_clip = request.FILES.get('video_clip')
        video_thumbnail = request.FILES.get('video_thumbnail')
        
        if not video_clip:
            print("No video clip in request")
            return Response({'error': 'No video clip provided'}, status=400)
            
        if not video_thumbnail:
            print("No thumbnail in request")
            # Continue without thumbnail
        
        # Create the alert
        from .models import ShopliftingAlert
        alert = ShopliftingAlert.objects.create(
            camera=camera,
            video_clip=video_clip,
            video_thumbnail=video_thumbnail
        )
        
        print(f"Alert created with ID {alert.id}")
        return Response({
            'status': 'success',
            'alert_id': alert.id,
            'message': 'Shoplifting alert created successfully'
        }, status=201)
        
    except Exception as e:
        import traceback
        print(f"Error creating alert: {e}")
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def get_shoplifting_alerts(request):
    try:
        # Get parameters
        is_reviewed = request.query_params.get('is_reviewed')
        
        # Start with all alerts for user's cameras
        from .models import ShopliftingAlert
        alerts_query = ShopliftingAlert.objects.filter(
            camera__user=request.user
        )
        
        # Filter by reviewed status if specified
        if is_reviewed is not None:
            is_reviewed = is_reviewed.lower() == 'true'
            alerts_query = alerts_query.filter(is_reviewed=is_reviewed)
            
        # Get alerts
        alerts = alerts_query.select_related('camera').order_by('-timestamp')
        
        # Format response
        alert_data = []
        for alert in alerts:
            alert_data.append({
                'id': alert.id,
                'camera_id': alert.camera.camera_id,
                'camera_name': alert.camera.name or f"Camera {alert.camera.id}",
                'timestamp': alert.timestamp,
                'is_reviewed': alert.is_reviewed,
                'video_ready': alert.video_ready,
                'video_clip': request.build_absolute_uri(alert.video_clip.url) if alert.video_clip else None,
                'thumbnail': request.build_absolute_uri(alert.video_thumbnail.url) if alert.video_thumbnail else None,
            })
            
        return Response({
            'status': 'success',
            'alerts': alert_data
        })
        
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def mark_alert_as_reviewed(request, alert_id):
    try:
        from .models import ShopliftingAlert
        
        # Get alert
        try:
            alert = ShopliftingAlert.objects.get(
                id=alert_id,
                camera__user=request.user
            )
        except ShopliftingAlert.DoesNotExist:
            return Response({'error': 'Alert not found'}, status=404)
            
        # Store file paths before we mark as reviewed
        video_path = alert.video_clip.path if alert.video_clip and hasattr(alert.video_clip, 'path') else None
        thumbnail_path = alert.video_thumbnail.path if alert.video_thumbnail and hasattr(alert.video_thumbnail, 'path') else None
        
        # Mark as reviewed in the model (which should update database but might not delete files)
        alert.mark_as_reviewed()
        
        # Explicitly delete the actual files from storage
        import os
        deleted_files = []
        
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                deleted_files.append('video')
                print(f"Deleted video file: {video_path}")
            except Exception as e:
                print(f"Error deleting video file: {str(e)}")
        
        if thumbnail_path and os.path.exists(thumbnail_path):
            try:
                os.remove(thumbnail_path)
                deleted_files.append('thumbnail')
                print(f"Deleted thumbnail file: {thumbnail_path}")
            except Exception as e:
                print(f"Error deleting thumbnail file: {str(e)}")
                
        # Also delete the alert record from the database to ensure it's completely gone
        try:
            alert.delete()
            print(f"Alert {alert_id} completely deleted from database")
        except Exception as e:
            print(f"Error deleting alert from database: {str(e)}")
        
        # Return clear success response
        return Response({
            'status': 'success',
            'message': 'Alert marked as reviewed and deleted successfully',
            'deleted_files': deleted_files,
            'deleted_alert_id': alert_id
        })
        
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['DELETE'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def delete_shoplifting_alert(request, alert_id):
    try:
        from .models import ShopliftingAlert
        
        # Get alert
        try:
            alert = ShopliftingAlert.objects.get(
                id=alert_id,
                camera__user=request.user
            )
        except ShopliftingAlert.DoesNotExist:
            return Response({'error': 'Alert not found'}, status=404)
            
        # Delete the alert
        alert.delete()
        
        return Response({
            'status': 'success',
            'message': 'Alert deleted'
        })
        
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def check_for_alerts(request):
    """API endpoint to check for unreviewed alerts - for real-time notification"""
    try:
        print(f"Checking for alerts for user {request.user.username}")
        from .models import ShopliftingAlert
        
        # Count unreviewed alerts
        unreviewed_count = ShopliftingAlert.objects.filter(
            camera__user=request.user,
            is_reviewed=False,
            camera__is_active=True
        ).count()
        
        print(f"Found {unreviewed_count} unreviewed alerts")
        
        # Get the most recent alert if available
        latest_alert = None
        if unreviewed_count > 0:
            alert = ShopliftingAlert.objects.filter(
                camera__user=request.user,
                is_reviewed=False,
                camera__is_active=True
            ).select_related('camera').order_by('-timestamp').first()
            
            if alert:
                print(f"Latest alert: id={alert.id}, camera={alert.camera.name}, time={alert.timestamp}")
                # Include video clip URL to help with debugging
                video_url = None
                thumbnail_url = None
                
                if alert.video_clip:
                    video_url = request.build_absolute_uri(alert.video_clip.url)
                
                if alert.video_thumbnail:
                    thumbnail_url = request.build_absolute_uri(alert.video_thumbnail.url)
                
                latest_alert = {
                    'id': alert.id,
                    'camera_id': alert.camera.camera_id,
                    'camera_name': alert.camera.name or f"Camera {alert.camera.id}",
                    'timestamp': alert.timestamp,
                    'timestamp_str': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'thumbnail': thumbnail_url,
                    'video_clip': video_url, 
                }
                
                # Log details for debugging
                print(f"Alert details: thumbnail={bool(thumbnail_url)}, video={bool(video_url)}")
        
        response_data = {
            'status': 'success',
            'unreviewed_count': unreviewed_count,
            'latest_alert': latest_alert
        }
        print(f"Returning response: {response_data}")
        return Response(response_data)
        
    except Exception as e:
        print(f"Error checking for alerts: {str(e)}")
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def set_test_mode(request, camera_id):
    """Set a camera detector to test mode for easier alert triggering"""
    try:
        lower_thresholds = request.data.get('lower_thresholds', True)
        
        with detector_lock:
            if camera_id in detector_instances:
                detector = detector_instances[camera_id]
                result = detector.set_test_mode(True, lower_thresholds)
                return Response({
                    'status': 'success',
                    'message': 'Test mode enabled',
                    'result': result
                })
            else:
                return Response({
                    'status': 'error',
                    'message': 'Camera not found'
                }, status=404)
    except Exception as e:
        return Response({
            'status': 'error',
            'message': str(e)
        }, status=500)

@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def force_trigger_alert(request, camera_id):
    """Force trigger an alert for testing purposes"""
    try:
        print(f"Attempting to force trigger alert for camera {camera_id}")
        
        # Check if camera exists and belongs to user
        try:
            camera = Camera.objects.get(
                camera_id=camera_id,
                user=request.user,
                camera_type='shoplift'
            )
        except Camera.DoesNotExist:
            return Response({
                'status': 'error',
                'message': f'Camera not found with ID {camera_id}'
            }, status=404)
        
        # Get token from request
        token = request.META.get('HTTP_AUTHORIZATION', '').split(' ')[1] if 'HTTP_AUTHORIZATION' in request.META else None
        if not token:
            # Generate a new token if not available in header
            refresh = RefreshToken.for_user(request.user)
            token = str(refresh.access_token)
        
        with detector_lock:
            if camera_id in detector_instances:
                detector = detector_instances[camera_id]
                
                # Update auth token
                if detector.auth_token != token:
                    print(f"Updating auth token for camera {camera_id}")
                    detector.auth_token = token
                
                # First ensure test mode is enabled
                detector.set_test_mode(True, True)
                print(f"Enabled test mode for camera {camera_id}")
                
                # Then force trigger
                result = detector.force_trigger_alert()
                print(f"Force trigger result: {result}")
                
                if result:
                    return Response({
                        'status': 'success',
                        'message': 'Alert triggered successfully',
                        'result': result
                    })
                else:
                    return Response({
                        'status': 'warning',
                        'message': 'Alert creation attempted but may have failed. Check console for details.'
                    })
            else:
                print(f"Camera {camera_id} not found in detector instances, trying to start it")
                # Try to initialize the detector
                detector = ShopliftDetector()
                if detector.start_detection(camera.video_path, camera_id, token):
                    detector_instances[camera_id] = detector
                    
                    # Set test mode and wait a bit for frames to accumulate
                    detector.set_test_mode(True, True)
                    time.sleep(2)
                    
                    # Force trigger
                    result = detector.force_trigger_alert()
                    print(f"Force trigger result after initializing: {result}")
                    
                    if result:
                        return Response({
                            'status': 'success',
                            'message': 'Alert triggered successfully after initialization',
                            'result': result
                        })
                    else:
                        return Response({
                            'status': 'warning',
                            'message': 'Alert creation attempted but may have failed after initialization. Check console for details.'
                        })
                else:
                    available_cameras = list(detector_instances.keys())
                    return Response({
                        'status': 'error',
                        'message': f'Failed to initialize camera detector. Available cameras: {available_cameras}'
                    }, status=500)
    except Exception as e:
        import traceback
        print(f"Error forcing alert trigger: {str(e)}")
        traceback.print_exc()
        return Response({
            'status': 'error',
            'message': f'Error triggering alert: {str(e)}'
        }, status=500)

@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def shoplifting_in_progress(request):
    """
    Handle notifications for shoplifting that is currently being recorded
    Creates a placeholder alert with a status indicating recording is in progress
    """
    try:
        print("======= RECEIVED SHOPLIFTING IN PROGRESS NOTIFICATION =======")
        data = request.data
        camera_id = data.get('camera_id')
        status_value = data.get('status', 'recording_in_progress')
        thumbnail_data = data.get('thumbnail_data')
        
        print(f"Camera ID: {camera_id}")
        print(f"Status: {status_value}")
        print(f"Thumbnail data present: {bool(thumbnail_data)}")
        
        # Get the camera and verify it's active
        try:
            camera = Camera.objects.get(
                camera_id=camera_id, 
                is_active=True  # Ensure camera is active
            )
            print(f"Found camera: {camera.name} (ID: {camera.id})")
        except Camera.DoesNotExist:
            print(f"ERROR: Camera not found or inactive with ID {camera_id}")
            return Response({'error': 'Camera not found or has been deactivated'}, status=status.HTTP_404_NOT_FOUND)
        
        # Check if this camera is in the process of being stopped
        with detector_lock:
            detector = detector_instances.get(camera_id)
            if detector and hasattr(detector, 'is_stopping') and detector.is_stopping:
                print(f"Camera {camera_id} is in the process of being stopped, ignoring alert")
                return Response({'error': 'Camera is being deactivated'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Convert base64 thumbnail data to file
        if thumbnail_data:
            try:
                thumbnail_bytes = base64.b64decode(thumbnail_data)
                print(f"Decoded thumbnail data: {len(thumbnail_bytes)} bytes")
                
                # Convert bytes to numpy array for OpenCV processing
                thumbnail_arr = np.frombuffer(thumbnail_bytes, np.uint8)
                thumbnail_img = cv2.imdecode(thumbnail_arr, cv2.IMREAD_COLOR)
                
                # Add text overlay - "RECORDING IN PROGRESS"
                h, w = thumbnail_img.shape[:2]
                overlay = thumbnail_img.copy()
                alpha = 0.5
                
                # Dark semi-transparent overlay in the middle
                cv2.rectangle(overlay, (0, h//2-30), (w, h//2+30), (0, 0, 0), -1)
                cv2.addWeighted(overlay, alpha, thumbnail_img, 1-alpha, 0, thumbnail_img)
                
                # Add text
                cv2.putText(thumbnail_img, 
                          "RECORDING IN PROGRESS", 
                          (w//2-160, h//2+10),
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          1.0, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Encode back to bytes
                _, thumb_buf = cv2.imencode('.jpg', thumbnail_img)
                thumbnail_bytes = thumb_buf.tobytes()
                
                # Create a new ShopliftingAlert with in-progress status
                from .models import ShopliftingAlert
                from django.core.files.base import ContentFile
                
                # Check if there's already an in-progress alert for this camera
                existing_alert = ShopliftingAlert.objects.filter(
                    camera=camera, 
                    status='recording_in_progress'
                ).first()
                
                if existing_alert:
                    print(f"Updating existing alert ID: {existing_alert.id}")
                    # Update existing in-progress alert
                    existing_alert.timestamp = timezone.now()  # Update timestamp
                    
                    # Update thumbnail
                    try:
                        thumbnail_name = f'in_progress_{camera_id}_{int(time.time())}.jpg'
                        existing_alert.video_thumbnail.save(
                            thumbnail_name, 
                            ContentFile(thumbnail_bytes),
                            save=True
                        )
                        print(f"Updated thumbnail: {thumbnail_name}")
                    except Exception as thumb_err:
                        print(f"Error saving thumbnail: {thumb_err}")
                    
                    return Response({
                        'status': 'success', 
                        'message': 'In-progress alert updated',
                        'id': existing_alert.id
                    }, status=status.HTTP_200_OK)
                else:
                    print("Creating new in-progress alert")
                    # Create new in-progress alert
                    alert = ShopliftingAlert(
                        camera=camera,
                        status='recording_in_progress',
                        video_ready=False
                    )
                    
                    # Save first to get an ID
                    alert.save()
                    print(f"Created new alert with ID: {alert.id}")
                    
                    # Then add the thumbnail
                    try:
                        thumbnail_name = f'in_progress_{camera_id}_{int(time.time())}.jpg'
                        alert.video_thumbnail.save(
                            thumbnail_name, 
                            ContentFile(thumbnail_bytes),
                            save=True
                        )
                        print(f"Saved thumbnail: {thumbnail_name}")
                    except Exception as thumb_err:
                        print(f"Error saving thumbnail: {thumb_err}")
                    
                    return Response({
                        'status': 'success', 
                        'message': 'In-progress alert created',
                        'id': alert.id
                    }, status=status.HTTP_201_CREATED)
            except Exception as decode_error:
                print(f"Error decoding thumbnail data: {decode_error}")
                import traceback
                traceback.print_exc()
                return Response({'error': f'Error processing thumbnail: {str(decode_error)}'}, 
                               status=status.HTTP_400_BAD_REQUEST)
        else:
            print("ERROR: No thumbnail data provided")
            return Response({'error': 'Thumbnail data is required'}, status=status.HTTP_400_BAD_REQUEST)
            
    except Exception as e:
        print(f"ERROR in shoplifting-in-progress: {e}")
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def update_alert_status(request):
    """
    Update the status of an existing shoplifting alert
    Used to update in-progress alerts with new information
    """
    try:
        data = request.data
        alert_id = data.get('alert_id')
        status_value = data.get('status')
        
        if not alert_id or not status_value:
            return Response({'error': 'Alert ID and status are required'}, status=status.HTTP_400_BAD_REQUEST)
            
        # Get the alert
        try:
            from .models import ShopliftingAlert
            alert = ShopliftingAlert.objects.get(id=alert_id)
        except ShopliftingAlert.DoesNotExist:
            return Response({'error': 'Alert not found'}, status=status.HTTP_404_NOT_FOUND)
            
        # Update the status
        alert.status = status_value
        
        # If completing the alert, update timestamp
        if status_value == 'completed':
            alert.created_at = timezone.now()
            
        alert.save()
        
        return Response({'status': 'success', 'message': 'Alert status updated'}, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def update_alert_evidence(request):
    """Update an existing alert with video evidence"""
    try:
        alert_id = request.data.get('alert_id')
        status_value = request.data.get('status', 'completed')
        video_clip = request.FILES.get('video_clip')
        video_thumbnail = request.FILES.get('video_thumbnail')

        if not alert_id:
            return Response({'error': 'Alert ID is required'}, status=400)

        try:
            alert = ShopliftingAlert.objects.get(id=alert_id)
        except ShopliftingAlert.DoesNotExist:
            return Response({'error': 'Alert not found'}, status=404)

        # If a new thumbnail is provided, use it
        if video_thumbnail:
            alert.video_thumbnail = video_thumbnail
        # Otherwise, if we have an existing thumbnail, remove the "RECORDING IN PROGRESS" text
        elif alert.video_thumbnail:
            try:
                # Get the current thumbnail
                thumbnail_path = alert.video_thumbnail.path
                
                # Read the image with OpenCV
                thumbnail_img = cv2.imread(thumbnail_path)
                if thumbnail_img is not None:
                    # Create clean version without overlay text
                    # We'll just use the original without the text overlay
                    # Simply save it back using the same path
                    _, thumb_buf = cv2.imencode('.jpg', thumbnail_img)
                    with open(thumbnail_path, 'wb') as f:
                        f.write(thumb_buf.tobytes())
            except Exception as e:
                print(f"Error removing 'recording in progress' text from thumbnail: {e}")
                # Continue even if this fails

        # Update the alert with the video evidence
        if video_clip:
            alert.video_clip = video_clip
            alert.video_ready = True

        # Update the status
        alert.status = status_value
        alert.save()

        # Add a system notification that evidence recording is complete
        response_data = {
            'status': 'success',
            'message': 'Alert evidence updated successfully',
            'alert_id': alert.id,
            'notification': {
                'title': 'Evidence Recording Complete',
                'message': f'Video evidence for alert #{alert.id} has been successfully saved.',
                'type': 'success'
            }
        }

        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        print(f"Error updating alert evidence: {e}")
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def check_camera_status(request, camera_id):
    """Check if a camera is still active - used by detector before sending alerts"""
    try:
        camera = Camera.objects.filter(
            camera_id=camera_id,
            is_active=True
        ).first()
        
        if camera:
            # Camera exists and is active
            return Response({
                'status': 'active',
                'camera_id': camera_id
            })
        else:
            # Camera is inactive or doesn't exist
            print(f"Camera status check: Camera {camera_id} is inactive or doesn't exist")
            return Response({
                'status': 'inactive',
                'camera_id': camera_id
            }, status=404)
    except Exception as e:
        print(f"Error checking camera status: {str(e)}")
        return Response({
            'status': 'error',
            'message': str(e)
        }, status=500)
