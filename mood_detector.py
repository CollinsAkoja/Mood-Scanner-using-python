import cv2
from deepface import DeepFace
import numpy as np

def map_emotion_to_status(emotion):
    """
    Maps detected emotions to the specific food status messages requested.
    """
    emotion = emotion.lower()
    
    if emotion == "angry":
        return "Hungry 😠", (0, 0, 255)  # Red
    elif emotion == "happy":
        return "Filled up 😊", (0, 255, 0)  # Green
    elif emotion == "sad":
        return "Drank garri all day 😢", (255, 0, 0)  # Blue
    else:
        return "Scanning...", (255, 255, 255)  # White for neutral/other

def main():
    # Initialize the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Mood Scanner Started... Press 'x' to quit.")
    
    # Frame counter to optimize performance (AI analysis is heavy)
    # We analyze every 10th frame to keep the video smooth
    frame_count = 0
    analyze_frequency = 10
    
    # Variables to store the last known results so text doesn't flicker
    last_status = "Waiting..."
    last_gender = ""
    last_color = (255, 255, 255)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Only perform heavy AI analysis every 'analyze_frequency' frames
        if frame_count % analyze_frequency == 0:
            try:
                # DeepFace analyze returns a list of faces detected
                # We set enforce_detection=False to avoid errors if no face is found
                results = DeepFace.analyze(
                    img_path=frame, 
                    actions=["emotion", "gender"], 
                    enforce_detection=False, 
                    silent=True
                )
                
                # If a face is detected
                if isinstance(results, list) and len(results) > 0:
                    result = results[0]
                    
                    # Get Dominant Emotion
                    dominant_emotion = result["dominant_emotion"]
                    
                    # Get Gender (DeepFace returns 'Man' or 'Woman')
                    gender_raw = result['dominant_gender']
                    gender_display = "Male" if gender_raw == "Man" else "Female"
                    
                    # Map emotion to your custom messages
                    status_text, color = map_emotion_to_status(dominant_emotion)
                    
                    # Update stored variables
                    last_status = status_text
                    last_gender = gender_display
                    last_color = color
                else:
                    last_status = "No Face Detected"
                    last_gender = ""
                    last_color = (255, 255, 255)
                    
            except Exception as e:
                # In case of any AI error, keep the last known status
                print(f"Analysis warning: {e}")
                pass

        #  Drawing on the Frame 
        
        # Display Gender
        if last_gender:
            cv2.putText(frame, f"Gender: {last_gender}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display Mood/Status
        cv2.putText(frame, f"Status: {last_status}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, last_color, 2)
        
        # Add a border to indicate scanning area
        h, w, _ = frame.shape
        cv2.rectangle(frame, (50, 50), (w-50, h-50), (100, 100, 100), 2)

        # Show the result
        cv2.imshow('Mood & Food Scanner', frame)

        # Press 'x' to exit
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()