from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
from collections import defaultdict
import tensorflow as tf
from datetime import datetime
import threading
import base64
from threading import Lock
import os
import time
import requests
from django.conf import settings

class AgeGenderDetector:
    def __init__(self):
        # Basic memory cleanup
        tf.keras.backend.clear_session()
        
        # Thread safety
        self.lock = Lock()
        self.frame_data = None
        self.is_running = False
        self.thread = None
        
        # Load models
        models_dir = os.path.dirname(__file__)
        self.gender_model = load_model(os.path.join(models_dir, 'gender_detection.keras'), compile=False)
        self.age_model = cv2.dnn.readNet(
            os.path.join(models_dir, 'age_net.caffemodel'),
            os.path.join(models_dir, 'age_deploy.prototxt')
        )

        # Constants
        self.MODEL_MEAN_VALUES = [104, 117, 123]
        self.gender_classes = ['man', 'woman']
        self.age_classifications = ['(0-3)', '(4-7)', '(8-12)', '(13-20)', '(21-32)', '(33-43)', '(44-53)', '(60-100)']
        
        # Tracking variables
        self.known_faces = {}
        self.confirmed_genders = {}
        self.confirmed_ages = {}
        self.next_face_id = 0
        self.face_confidences = {}
        self.timestamps = {}
        self.available_ids = []
        self.prediction_history = defaultdict(lambda: {'gender': [], 'age': []})
        self.data_analysis = []
        
        # Configuration
        self.CONFIDENCE_THRESHOLD = 0.8
        self.MAX_TRACKING_DISTANCE = 100
        self.TRACKING_MEMORY = 20
        
        # Analysis metrics
        self.camera_id = None
        self.auth_token = None
        self.last_analysis_time = datetime.now()
        self.male_count = 0
        self.female_count = 0
        self.age_counts = {
            '(0-3)': 0,
            '(4-7)': 0,
            '(8-12)': 0,
            '(13-20)': 0,
            '(21-32)': 0,
            '(33-43)': 0,
            '(44-53)': 0,
            '(60-100)': 0
        }
        
    def calculate_distance(self, pos1, pos2):
        center1 = ((pos1[0] + pos1[2]) // 2, (pos1[1] + pos1[3]) // 2)
        center2 = ((pos2[0] + pos2[2]) // 2, (pos2[1] + pos2[3]) // 2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def send_analysis_data(self):
        """Send detection data to analysis endpoint"""
        if not self.camera_id or not self.auth_token:
            return
            
        current_time = datetime.now()
        if (current_time - self.last_analysis_time).total_seconds() >= 10:
            try:
                headers = {
                    'Authorization': f'Bearer {self.auth_token}',
                    'Content-Type': 'application/json'
                }
                
                # Map age classifications to database field names
                age_data = {
                    'age_0_3_count': self.age_counts['(0-3)'],
                    'age_4_7_count': self.age_counts['(4-7)'],
                    'age_8_12_count': self.age_counts['(8-12)'],
                    'age_13_20_count': self.age_counts['(13-20)'],
                    'age_21_32_count': self.age_counts['(21-32)'],
                    'age_33_43_count': self.age_counts['(33-43)'],
                    'age_44_53_count': self.age_counts['(44-53)'],
                    'age_60_100_count': self.age_counts['(60-100)']
                }
                
                data = {
                    'camera_id': self.camera_id,
                    'male_count': self.male_count,
                    'female_count': self.female_count,
                    **age_data
                }
                
                response = requests.post(
                    f'{settings.API_URL}/api/update-age-gender-data/',
                    json=data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    # Reset counters after successful update
                    self.male_count = 0
                    self.female_count = 0
                    self.age_counts = {age: 0 for age in self.age_counts}
                    self.last_analysis_time = current_time
                    
            except Exception as e:
                print(f"Error sending analysis data: {e}")

    def start_detection(self, video_path, camera_id=None, auth_token=None):
        try:
            if video_path.lower() == 'webcam' or video_path == '0':
                video_path = 0  # Use default webcam
                
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video source: {video_path}")
                return False
                
            # Set buffer size to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            with self.lock:
                if not self.is_running:
                    self.camera_id = camera_id
                    self.auth_token = auth_token
                    self.is_running = True
                    self.thread = threading.Thread(target=self._detection_loop, args=(video_path,))
                    self.thread.daemon = True
                    self.thread.start()
                    return True
                    
            if cap is not None:
                cap.release()
            return False
            
        except Exception as e:
            print(f"Error starting detection: {e}")
            if cap is not None:
                cap.release()
            return False

    def stop_detection(self):
        with self.lock:
            if self.is_running:
                self.is_running = False
                if self.thread:
                    self.thread.join(timeout=1.0)
                    self.thread = None
                return True
        return False

    def get_frame(self):
        with self.lock:
            return self.frame_data

    def _detection_loop(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video source")

        frame_counter = 0
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(video_path, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video file
                        continue
                    else:
                        cap.release()  # Release current webcam handle
                        time.sleep(1)  # Wait before reconnecting
                        cap = cv2.VideoCapture(video_path)  # Reopen webcam
                        continue

                try:
                    # Process frame
                    faces, confidences = cv.detect_face(frame)
                    
                    # Draw rectangles for detected faces
                    for face, confidence in zip(faces, confidences):
                        (startX, startY, endX, endY) = face
                        face_crop = frame[startY:endY, startX:endX]
                        
                        # Gender detection
                        gender_crop = cv2.resize(face_crop, (96, 96))
                        gender_crop = gender_crop.astype("float") / 255.0
                        gender_crop = img_to_array(gender_crop)
                        gender_crop = np.expand_dims(gender_crop, axis=0)
                        gender_pred = self.gender_model.predict(gender_crop, verbose=0)[0]
                        gender = self.gender_classes[np.argmax(gender_pred)]

                        # Update gender counts
                        if gender == 'man':
                            self.male_count += 1
                        else:
                            self.female_count += 1
                        
                        # Age detection
                        face_blob = cv2.dnn.blobFromImage(cv2.resize(face_crop, (227, 227)), 
                                                        1.0, (227, 227),
                                                        self.MODEL_MEAN_VALUES, swapRB=True)
                        self.age_model.setInput(face_blob)
                        age_pred = self.age_model.forward()
                        age = self.age_classifications[age_pred[0].argmax()]
                        
                        # Update age counts
                        self.age_counts[age] += 1
                        
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                        # Display gender and age on the frame
                        label = f"{gender}, {age}"
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.putText(frame, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Encode frame regardless of face detection
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    with self.lock:
                        self.frame_data = frame_base64
                        
                    # Send analysis data
                    self.send_analysis_data()

                except Exception as inner_e:
                    print(f"Error processing frame: {inner_e}")
                    continue

                frame_counter += 1
                time.sleep(0.01)  # Small delay to prevent CPU overload

        except Exception as e:
            print(f"Error in detection loop: {e}")
        finally:
            if cap is not None and cap.isOpened():
                cap.release()