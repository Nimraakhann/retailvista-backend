import requests
import base64
import cv2
import numpy as np
import json
import sys
import os
import time

def send_test_notification(camera_id, token=None):
    """Send a test notification to the shoplifting-in-progress endpoint"""
    
    print(f"Sending test notification for camera {camera_id}")
    
    # Create a test image with text
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)  # Dark gray background
    
    # Add text
    cv2.putText(
        img,
        "TEST NOTIFICATION",
        (100, 240),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        (255, 255, 255),
        2
    )
    
    cv2.putText(
        img,
        f"Camera: {camera_id}",
        (100, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 255),
        1
    )
    
    cv2.putText(
        img,
        f"Time: {time.strftime('%H:%M:%S')}",
        (100, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 255, 200),
        1
    )
    
    # Add red recording indicator
    cv2.circle(
        img,
        (50, 50),
        10,
        (0, 0, 255),
        -1
    )
    
    # Save image for reference
    debug_dir = "debug_images"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    cv2.imwrite(os.path.join(debug_dir, f"test_notification_{camera_id}.jpg"), img)
    
    # Convert to base64
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Prepare request data
    data = {
        'camera_id': camera_id,
        'status': 'recording_in_progress',
        'thumbnail_data': img_base64
    }
    
    # Prepare headers
    headers = {
        'Content-Type': 'application/json'
    }
    
    if token:
        headers['Authorization'] = f'Bearer {token}'
    
    # Send request
    print("Sending request to http://localhost:8000/api/shoplifting-in-progress/")
    try:
        response = requests.post(
            'http://localhost:8000/api/shoplifting-in-progress/',
            json=data,
            headers=headers,
            timeout=10
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text}")
        
        if response.status_code in [200, 201]:
            print("SUCCESS! Notification sent.")
            try:
                print(f"Response data: {json.dumps(response.json(), indent=2)}")
            except:
                pass
        else:
            print(f"ERROR: Failed to send notification. Status: {response.status_code}")
            
    except Exception as e:
        print(f"ERROR: Exception occurred while sending notification: {e}")

if __name__ == "__main__":
    # Get camera ID from command line argument or use default
    camera_id = sys.argv[1] if len(sys.argv) > 1 else None
    token = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not camera_id:
        print("ERROR: Please provide a camera ID as the first argument")
        print("Usage: python test_notification.py <camera_id> [<token>]")
        sys.exit(1)
    
    send_test_notification(camera_id, token) 