# Camera Packages
import cv2
import time
import os
from picamera2 import Picamera2
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, Response, send_file
import logging
import socket
import netifaces

# Stop Flask Debugging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# AI Packages
from google import genai
from google.genai import types

# TTS Packages
from gtts import gTTS
from mutagen.mp3 import MP3
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GemApiKey"))

askAi = "Identify the main object in the image with a single word. If a hand is holding an object, identify only the held object. If no clear object is present or identifiable, respond with nothing."

# Flask App Setup
app = Flask(__name__)

# Global State
current_status = "Initializing..."
last_object = "Waiting..."
state_lock = threading.Lock()

def update_status(status):
    global current_status
    with state_lock:
        current_status = status
    print(f"[STATUS] {status}")

def update_object(obj_name):
    global last_object
    with state_lock:
        last_object = obj_name
    print(f"[OBJECT] {obj_name}")

def sendImageToAI(image_bytes):
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            askAi, # Text Prompt
            types.Part.from_bytes(
                data=image_bytes, mime_type="image/jpeg" # Image
            )
        ]
    )
    return response.text

def say(text):
    update_status(f"Playing Audio: {text}")
    language = 'en'
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("output.mp3")
    audio = MP3("output.mp3")
    print("Playing Audio.")
    os.system("mpg123 -q output.mp3")
    time.sleep(audio.info.length + 1)
    print("-" * 20)

def motion_detection(newFrame, oldFrame, frame):
    # Calculate Difference for Previous Frame
    frameDifference = cv2.absdiff(oldFrame, newFrame)
    # Threshold the frame to binarize it for Contour Detection
    thresh = cv2.threshold(frameDifference, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    # Find Contours (Shapes of the moving objects)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_found = False
    # Check if motion is detected
    for contour in contours:
        # If the contour is too small, ignore it
        if cv2.contourArea(contour) < 500:
            continue
        
        # Draw the contour on the frame
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_found = True
    
    if motion_found:
        cv2.imwrite('motion.jpg', frame)
        return True
    
    return False

def run_camera_loop():
    # Setup Camera
    picam2 = Picamera2(camera_num=0)
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    print("\n\n")
    update_status("Camera started. Monitoring...")

    backgroundFrame = None
    prevFrame = None
    
    lastObject = ""

    motionActive = False
    lastMotionTime = 0.0

    while True:
        # Delay
        time.sleep(0.1)

        # Capture frame
        frame = picam2.capture_array()
        if frame is None:
            break
        
        # Flip the frame 180 degrees (camera is upside down)
        frame = cv2.flip(frame, -1)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur Image
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if backgroundFrame is None:
            print("open http://" + netifaces.ifaddresses("wlan0")[netifaces.AF_INET][0]['addr'] + ":5000")
            print("Background set.\n" + "-" * 20)
            backgroundFrame = gray
            prevFrame = gray
            continue

        if prevFrame is None:
            prevFrame = gray
            continue
        
        # Check for ACTIVE motion
        currentMotionFound = motion_detection(gray, prevFrame, frame)
        
        # If motion is detected
        if currentMotionFound:
            motionActive = True
            # Save last motion time
            lastMotionTime = time.time()
        else:
            # No motion in frame
            if motionActive:
                # Check if motion has stopped for 1 second
                timeSinceMotion = time.time() - lastMotionTime
                if timeSinceMotion > 1.0:
                    # Check if object is still there
                    objectPresent = motion_detection(gray, backgroundFrame, frame)
                    
                    if objectPresent:
                        update_status("Sending to AI...")
                        _, buffer = cv2.imencode('.jpg', frame)
                        text = sendImageToAI(buffer.tobytes())

                        update_status(f"AI Response: {text}")
                        update_object(text)

                        if  text != lastObject:
                            say(text)
                            lastObject = text
                        elif text == "nothing":
                            print("No Object is Shown.")
                            print("-" * 20)
                        else:
                            print("Same Object is Shown.")
                            print("-" * 20)

                    update_status("Monitoring...")
                    time.sleep(1)

                    # Reset motion state
                    motionActive = False

        # Update previous frame
        prevFrame = gray

    picam2.stop()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def get_status():
    with state_lock:
        return jsonify({"status": current_status, "object": last_object})

@app.route('/image')
def get_image():
    if os.path.exists('motion.jpg'):
        return send_file('motion.jpg', mimetype='image/jpeg')
    else:
        return "No image available", 404

def main():
    # Start camera loop in a separate thread
    camera_thread = threading.Thread(target=run_camera_loop, daemon=True)
    camera_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()