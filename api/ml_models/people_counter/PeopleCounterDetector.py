from ultralytics import YOLO
import cv2
import torch
import gc
import numpy as np  
import os
import threading
import base64
from threading import Lock
import time
from api.models import Camera

class PeopleCounterDetector:
    def __init__(self, camera_id):
        # Memory optimization
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        # Initialize CUDA settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)

        # Display settings
        self.target_width = 1020
        self.target_height = 600

        # Thread safety
        self.lock = Lock()
        self.frame_data = None
        self.thread = None
        
        # Database reference
        self.camera_id = camera_id
        self.camera = Camera.objects.get(camera_id=camera_id, camera_type='people_counter')

        # Load YOLO model
        model_path = os.path.join(os.path.dirname(__file__), "yolo11n.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Tracking variables
        self.person_states = {}  # track_id: {'entered': bool, 'counted': bool}
        self.entry_points = self.camera.entry_points
        self.exit_points = self.camera.exit_points
        self.num_areas = len(self.entry_points)
        self.frame_copy = None

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            cv2.circle(self.frame_copy, (x, y), 5, (0, 255, 0), -1)

    def setup_counting_zones(self, first_frame, num_areas=1):
        """Set up entry and exit zones for people counting"""
        self.num_areas = num_areas
        return True  # Just return True, all points will come from frontend

    def start_detection(self, video_path, entry_points=None, exit_points=None):
        if self.camera.is_running:
            return False
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video source")
        
        # Add debug logging
        print("Received points:", {
            "entry_points": entry_points,
            "exit_points": exit_points
        })
        
        if entry_points and exit_points:
            # Validate the points
            if not isinstance(entry_points, list) or not isinstance(exit_points, list):
                print("Error: entry_points and exit_points must be lists")
                return False
                
            if len(entry_points) < 3 or len(exit_points) < 3:
                print("Error: entry_points and exit_points must have at least 3 points for a valid polygon")
                return False
                
            # Convert points for internal storage
            self.entry_points = [entry_points]  # Wrap in list for multiple areas
            self.exit_points = [exit_points]    # Wrap in list for multiple areas
            
            print("Converted points:", {
                "entry_points": self.entry_points,
                "exit_points": self.exit_points
            })
            
            # Save to the database
            self.camera.entry_points = self.entry_points
            self.camera.exit_points = self.exit_points
            self.camera.save()
        else:
            # Try to use existing points from the database if available
            if (self.camera.entry_points and self.camera.exit_points and 
                len(self.camera.entry_points) > 0 and len(self.camera.exit_points) > 0):
                self.entry_points = self.camera.entry_points
                self.exit_points = self.camera.exit_points
                print("Using existing points from database:", {
                    "entry_points": self.entry_points,
                    "exit_points": self.exit_points
                })
            else:
                print("Error: No entry and exit points provided or found in database")
                return False
        
        cap.release()
        
        self.camera.is_running = True
        self.camera.save()
        self.thread = threading.Thread(target=self._detection_loop, args=(video_path,))
        self.thread.daemon = True
        self.thread.start()
        return True

    def stop_detection(self):
        self.camera.is_running = False
        self.camera.save()
        if self.thread:
            self.thread.join()
            self.thread = None

    def get_frame(self):
        with self.lock:
            return self.frame_data

    def _detection_loop(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video source")
        
        frame_count = 0
        last_log_time = time.time()
        last_entry_count = 0
        
        try:
            while self.camera.is_running:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame_count += 1
                if frame_count % 2 != 0:  # Process every other frame
                    continue

                try:
                    # Resize frame
                    frame = cv2.resize(frame, (self.target_width, self.target_height))

                    # Check if entry_points and exit_points are valid before using them
                    if (self.entry_points and self.exit_points and 
                        len(self.entry_points) > 0 and len(self.exit_points) > 0 and
                        isinstance(self.entry_points[0], list) and isinstance(self.exit_points[0], list) and 
                        len(self.entry_points[0]) > 2 and len(self.exit_points[0]) > 2):
                        
                        # Convert points to numpy arrays for polygon operations
                        entry_points = np.array(self.entry_points[0], dtype=np.int32)
                        exit_points = np.array(self.exit_points[0], dtype=np.int32)
                        
                        # Draw polygons
                        cv2.polylines(frame, [entry_points], True, (0, 255, 0), 2)
                        cv2.polylines(frame, [exit_points], True, (255, 0, 0), 2)
    
                        # Run detection and tracking
                        results = self.model.track(frame, persist=True, verbose=False)
    
                        if results[0].boxes is not None and results[0].boxes.id is not None:
                            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
    
                            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                                if 'person' in self.model.names[class_id]:
                                    x1, y1, x2, y2 = box
                                    center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
                                    # Initialize person state if new
                                    if track_id not in self.person_states:
                                        self.person_states[track_id] = {'entered': False, 'counted': False}
    
                                    # Now use the numpy array for pointPolygonTest
                                    if cv2.pointPolygonTest(entry_points, center_point, False) >= 0:
                                        self.person_states[track_id]['entered'] = True
    
                                    if self.person_states[track_id]['entered'] and not self.person_states[track_id]['counted']:
                                        if cv2.pointPolygonTest(exit_points, center_point, False) >= 0:
                                            self.camera.entry_count += 1
                                            self.camera.save()
                                            self.person_states[track_id]['counted'] = True
    
                                    # Draw bounding box and ID
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    cv2.putText(frame, f'{track_id}', (x1, y1-10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        # If points are not valid, display a message on the frame
                        cv2.putText(frame, "Entry/Exit zones not configured", (20, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        print("Invalid entry or exit points:", self.entry_points, self.exit_points)

                    # Add count to frame
                    cv2.putText(frame, f'Entry Count: {self.camera.entry_count}', 
                              (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Encode frame
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    with self.lock:
                        self.frame_data = frame_base64
                    
                    # Log data every 30 seconds for analysis purposes
                    current_time = time.time()
                    if current_time - last_log_time >= 30 and self.camera.entry_count > last_entry_count:
                        # Log new entries
                        try:
                            # Import here to avoid circular import
                            from api.models import PeopleCounterData
                            
                            # Calculate the difference since last log
                            new_entries = self.camera.entry_count - last_entry_count
                            
                            # Create a data point
                            PeopleCounterData.objects.create(
                                camera=self.camera,
                                entry_count=new_entries,
                                zone_name=self.camera.name or 'Main Zone'
                            )
                            
                            # Update tracking variables
                            last_log_time = current_time
                            last_entry_count = self.camera.entry_count
                        except Exception as e:
                            print(f"Error logging data for analysis: {e}")

                    time.sleep(0.01)  # Small delay to prevent CPU overload

                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue

        except Exception as e:
            print(f"Error in detection loop: {e}")
        finally:
            cap.release()

    def get_count(self):
        """Get the current entry count"""
        return self.camera.entry_count