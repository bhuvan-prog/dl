import cv2
import numpy as np
import tensorflow as tf
import pickle

def webcam_detection():
    # Try to load model from pickle, fall back to h5
    try:
        with open('mask_detector.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded from pickle file")
    except:
        try:
            model = tf.keras.models.load_model("mask_detector.h5")
            print("Model loaded from h5 file")
        except:
            print("Error: Could not load model. Please train the model first.")
            return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    labels_dict = {0: 'No Mask', 1: 'Mask'}
    color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
    
    cap = cv2.VideoCapture(0)
    
    print("Starting webcam detection. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            resized = cv2.resize(face_img, (120, 120))
            resized = resized / 255.0
            reshaped = np.reshape(resized, (1, 120, 120, 3))
            
            result = model.predict(reshaped)
            label = 1 if result[0][0] > 0.5 else 0
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color_dict[label], 2)
            cv2.putText(frame, labels_dict[label], (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_dict[label], 2)
        
        cv2.imshow("Face Mask Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_detection()
