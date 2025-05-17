from django.db import models
from django.contrib.auth.models import User
import uuid
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.utils import timezone
from datetime import timedelta
import numpy as np
import json
import os

class Map(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='maps')
    iframe_code = models.TextField()
    editor_link = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Map for {self.user.email}"

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=100, blank=True, null=True)
    company = models.CharField(max_length=100, blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True)
    email_verified = models.BooleanField(default=False)
    verification_token = models.CharField(max_length=100, blank=True)
    verification_token_expires = models.DateTimeField(null=True, blank=True)
    store_id = models.CharField(max_length=10, unique=True, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        if not self.store_id:
            # Generate a unique store ID if not already set
            self.store_id = f'S{uuid.uuid4().hex[:8].upper()}'
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.user.email}'s profile"

class Promotion(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='promotions')
    name = models.CharField(max_length=200)
    image = models.ImageField(upload_to='promotions/')
    status = models.CharField(
        max_length=20,
        choices=[('active', 'Active'), ('inactive', 'Inactive')],
        default='inactive'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} - {self.user.email}"

class Camera(models.Model):
    CAMERA_TYPES = [
        ('shoplift', 'Shoplifting Detection'),
        ('age_gender', 'Age & Gender Detection'),
        ('people_counter', 'People Counter')
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    camera_id = models.CharField(max_length=100)
    name = models.CharField(max_length=255, blank=True)  # Add this line
    video_path = models.CharField(max_length=255)
    camera_type = models.CharField(
        max_length=50,
        choices=CAMERA_TYPES,
        db_index=True
    )
    is_active = models.BooleanField(default=True)
    # Add people counter specific fields
    entry_count = models.IntegerField(default=0)
    entry_points = models.JSONField(default=list)
    exit_points = models.JSONField(default=list)
    is_running = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        unique_together = ['user', 'camera_id', 'camera_type']
        indexes = [
            models.Index(fields=['user', 'camera_type', 'is_active'])
        ]

    def save(self, *args, **kwargs):
        # Convert numpy arrays to lists before saving
        if isinstance(self.entry_points, np.ndarray):
            self.entry_points = self.entry_points.tolist()
        if isinstance(self.exit_points, np.ndarray):
            self.exit_points = self.exit_points.tolist()
            
        # Ensure entry_points and exit_points are properly formatted as lists
        if self.entry_points and not isinstance(self.entry_points, list):
            print(f"Warning: Converting entry_points type {type(self.entry_points)} to list")
            try:
                self.entry_points = list(self.entry_points)
            except:
                print(f"Error converting entry_points, setting to empty list")
                self.entry_points = []
                
        if self.exit_points and not isinstance(self.exit_points, list):
            print(f"Warning: Converting exit_points type {type(self.exit_points)} to list")
            try:
                self.exit_points = list(self.exit_points)
            except:
                print(f"Error converting exit_points, setting to empty list")
                self.exit_points = []
                
        super().save(*args, **kwargs)

class DetectionData(models.Model):
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='detection_data')
    timestamp = models.DateTimeField(auto_now_add=True)
    total_events = models.IntegerField(default=0)
    suspicious_events = models.IntegerField(default=0)
    
    class Meta:
        indexes = [
            models.Index(fields=['camera', 'timestamp']),
            models.Index(fields=['timestamp'])
        ]
        
    @classmethod
    def cleanup_old_data(cls):
        """Delete data older than 1 year"""
        from django.utils import timezone
        from datetime import timedelta
        one_year_ago = timezone.now() - timedelta(days=365)
        cls.objects.filter(timestamp__lt=one_year_ago).delete()

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        # Cleanup old data after each save
        self.cleanup_old_data()

class AgeGenderDetectionData(models.Model):
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='age_gender_data')
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Gender counts
    male_count = models.IntegerField(default=0)
    female_count = models.IntegerField(default=0)
    
    # Age group counts - match the model's age classifications
    age_0_3_count = models.IntegerField(default=0)
    age_4_7_count = models.IntegerField(default=0)
    age_8_12_count = models.IntegerField(default=0)
    age_13_20_count = models.IntegerField(default=0)
    age_21_32_count = models.IntegerField(default=0)
    age_33_43_count = models.IntegerField(default=0)
    age_44_53_count = models.IntegerField(default=0)
    age_60_100_count = models.IntegerField(default=0)
    
    total_detections = models.IntegerField(default=0)
    
    class Meta:
        indexes = [
            models.Index(fields=['camera', 'timestamp']),
            models.Index(fields=['timestamp'])
        ]
        
    @classmethod
    def cleanup_old_data(cls):
        """Delete data older than 1 year"""
        from django.utils import timezone
        from datetime import timedelta
        one_year_ago = timezone.now() - timedelta(days=365)
        cls.objects.filter(timestamp__lt=one_year_ago).delete()

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        # Cleanup old data after each save
        self.cleanup_old_data()

class PeopleCounterData(models.Model):
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='people_counter_data')
    timestamp = models.DateTimeField(auto_now_add=True)
    entry_count = models.IntegerField(default=0)
    zone_name = models.CharField(max_length=100, blank=True, default='Main Zone')
    
    class Meta:
        indexes = [
            models.Index(fields=['camera', 'timestamp']),
            models.Index(fields=['timestamp']),
            models.Index(fields=['zone_name'])
        ]
        
    @classmethod
    def cleanup_old_data(cls):
        """Delete data older than 1 year"""
        from django.utils import timezone
        from datetime import timedelta
        one_year_ago = timezone.now() - timedelta(days=365)
        cls.objects.filter(timestamp__lt=one_year_ago).delete()

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        # Cleanup old data after each save
        self.cleanup_old_data()

class ShopliftingAlert(models.Model):
    STATUS_CHOICES = [
        ('new', 'New'),
        ('recording_in_progress', 'Recording in Progress'),
        ('completed', 'Completed'),
        ('reviewed', 'Reviewed')
    ]
    
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='alerts')
    timestamp = models.DateTimeField(auto_now_add=True)
    video_clip = models.FileField(upload_to='shoplifting_alerts/', blank=True, null=True)
    video_thumbnail = models.ImageField(upload_to='shoplifting_thumbnails/', blank=True, null=True)
    is_reviewed = models.BooleanField(default=False)
    review_date = models.DateTimeField(blank=True, null=True)
    auto_delete_date = models.DateTimeField(blank=True, null=True)
    status = models.CharField(max_length=25, choices=STATUS_CHOICES, default='new')
    detection_metadata = models.JSONField(blank=True, null=True)
    video_ready = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['camera', 'timestamp']),
            models.Index(fields=['is_reviewed']),
            models.Index(fields=['auto_delete_date']),
        ]
    
    def save(self, *args, **kwargs):
        # Set auto-delete date to 15 days from creation if not already set
        if not self.auto_delete_date:
            self.auto_delete_date = timezone.now() + timezone.timedelta(days=15)
        
        # If marked as reviewed, set for immediate deletion
        if self.is_reviewed and not self.review_date:
            self.review_date = timezone.now()
            # Set auto_delete_date to now to mark for immediate deletion
            self.auto_delete_date = timezone.now()
            
        super().save(*args, **kwargs)
        # Cleanup old alerts after each save (like other models)
        self.cleanup_old_alerts()
    
    def mark_as_reviewed(self):
        self.is_reviewed = True
        self.review_date = timezone.now()
        # Set auto_delete_date to now to mark for immediate deletion
        self.auto_delete_date = timezone.now()
        self.save()
    
    @classmethod
    def cleanup_old_alerts(cls):
        """Delete alerts past their auto_delete_date"""
        now = timezone.now()
        cls.objects.filter(auto_delete_date__lt=now).delete()
