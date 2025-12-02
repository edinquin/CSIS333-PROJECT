# Camera Packages
import cv2
import time
import os
from picamera2 import Picamera2

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

def main():
    # Setup Camera
    picam2 = Picamera2(camera_num=0)
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    print("Camera started successfully.")

    backgroundFrame = None
    prevFrame = None
    lastObject = ""

    motionActive = False
    lastMotionTime = 0.0

    frameCount = 0

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
            print("Background set." + "\n" + "-" * 20)
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
                        print("Sending To Ai.")
                        _, buffer = cv2.imencode('.jpg', frame)
                        text = sendImageToAI(buffer.tobytes())
                        print("Ai Response: " + text)
                        if  text != lastObject:
                            say(text)
                            lastObject = text
                        elif text == "nothing":
                            print("No Object is Shown.")
                            print("-" * 20)
                        else:
                            print("Same Object is Shown.")
                            print("-" * 20)

                    time.sleep(1)

                    # Reset motion state
                    motionActive = False
        
        # Update previous frame
        prevFrame = gray

    picam2.stop()
            

if __name__ == "__main__":
    main()