from django.urls import path
from .views import (
    login_view, signup, verify_email, reset_password, 
    update_profile, contact_form, get_all_users, delete_user, admin_update_user,
    request_password_reset,
    verify_reset_token,
    set_new_password,
    promotion_list,
    active_promotions,
    promotion_detail,
    map_view,
    connect_people_counter,
    delete_people_counter_camera,
    get_people_counter_frame,
    get_people_counter_cameras,
    get_age_gender_analysis,
    update_age_gender_data,
    create_shoplifting_alert,
    get_shoplifting_alerts,
    mark_alert_as_reviewed,
    delete_shoplifting_alert,
    check_for_alerts,
    set_test_mode,
    force_trigger_alert,
    check_camera_status
)
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView
)
from . import views

urlpatterns = [
    path('login/', login_view, name='login'),
    path('signup/', signup, name='signup'),
    path('verify-email/', verify_email, name='verify_email'),
    path('reset-password/', reset_password, name='reset_password'),
    path('update-profile/', update_profile, name='update_profile'),
    path('contact/', contact_form, name='contact_form'),
    path('get_all_users/', get_all_users, name='get_all_users'),
    path('delete_user/<int:user_id>/', delete_user, name='delete_user'),
    path('verify-token/', TokenVerifyView.as_view(), name='token-verify'),
    
    # JWT Token URLs
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('admin/update_user/', admin_update_user, name='admin_update_user'),
    path('request-password-reset/', request_password_reset, name='request_password_reset'),
    path('verify-reset-token/', verify_reset_token, name='verify_reset_token'),
    path('set-new-password/', set_new_password, name='set_new_password'),
    path('promotions/', promotion_list, name='promotion-list'),
    path('promotions/active/', active_promotions, name='active-promotions'),
    path('promotions/<int:pk>/', promotion_detail, name='promotion-detail'),
    path('map/', map_view, name='map-view'),
    path('get-cameras/', views.get_cameras, name='get_cameras'),
    path('connect-camera/', views.connect_camera, name='connect_camera'),
    path('delete-camera/<str:camera_id>/', views.delete_camera, name='delete_camera'),
    path('get-frame/<str:camera_id>/', views.get_frame, name='get_frame'),
    path('connect-age-gender-camera/', views.connect_age_gender_camera, name='connect_age_gender_camera'),
    path('get-age-gender-frame/<str:camera_id>/', views.get_age_gender_frame, name='get_age_gender_frame'),
    path('delete-age-gender-camera/<str:camera_id>/', views.delete_age_gender_camera, name='delete_age_gender_camera'),
    path('get-age-gender-cameras/', views.get_age_gender_cameras, name='get_age_gender_cameras'),
    path('connect-people-counter/', connect_people_counter, name='connect_people_counter'),
    path('delete-people-counter-camera/<str:camera_id>/', delete_people_counter_camera, name='delete_people_counter_camera'),
    path('get-people-counter-frame/<str:camera_id>/', get_people_counter_frame, name='get_people_counter_frame'),
    path('get-people-counter-cameras/', get_people_counter_cameras, name='get_people_counter_cameras'),
    path('get-first-frame/', views.get_first_frame, name='get_first_frame'),
    path('analysis-data/', views.get_analysis_data, name='get_analysis_data'),
    path('update-detection-data/', views.update_detection_data, name='update_detection_data'),
    
    # New Age-Gender Analysis endpoints
    path('age-gender-analysis/', get_age_gender_analysis, name='get_age_gender_analysis'),
    path('update-age-gender-data/', update_age_gender_data, name='update_age_gender_data'),
    
    # New People Counter Analysis endpoints
    path('people-counter-analysis/', views.get_people_counter_analysis, name='get_people_counter_analysis'),
    path('update-people-counter-data/', views.update_people_counter_data, name='update_people_counter_data'),
    
    # Shoplifting Alert endpoints
    path('create-shoplifting-alert/', create_shoplifting_alert, name='create_shoplifting_alert'),
    path('shoplifting-alerts/', get_shoplifting_alerts, name='get_shoplifting_alerts'),
    path('mark-alert-as-reviewed/<int:alert_id>/', mark_alert_as_reviewed, name='mark_alert_as_reviewed'),
    path('delete-shoplifting-alert/<int:alert_id>/', delete_shoplifting_alert, name='delete_shoplifting_alert'),
    path('check-for-alerts/', check_for_alerts, name='check_for_alerts'),
    path('shoplifting-in-progress/', views.shoplifting_in_progress, name='shoplifting_in_progress'),
    path('update-alert-status/', views.update_alert_status, name='update_alert_status'),
    path('update-alert-evidence/', views.update_alert_evidence, name='update_alert_evidence'),
    path('check-camera-status/<str:camera_id>/', check_camera_status, name='check_camera_status'),
    
    # Testing endpoints
    path('set-test-mode/<str:camera_id>/', set_test_mode, name='set_test_mode'),
    path('force-trigger-alert/<str:camera_id>/', force_trigger_alert, name='force_trigger_alert'),
]
