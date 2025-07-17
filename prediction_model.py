#The Prediction Model
import cv2
import tensorflow as tf
import numpy as np

# Load and recompile the model to suppress warnings
model = tf.keras.models.load_model('M:/PROJECTS/AI_PROJECT/DATASET/DATASET/final_ai_model.h5')
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class_names = ['BALL_DEFECT', 'CAGE_DEFECT', 'HEALTHY', 'INNER_RACE_DEFECT', 'LACK_OF_LUBRICATION', 'OUTER_RACE_DEFECT']

def realtime_video_predict(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess frame
        img = cv2.resize(frame, (224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        pred = model.predict(img_array, verbose=0)
        confidence = np.max(pred)
        predicted_class = class_names[np.argmax(pred)]
        
        # Overlay prediction
        cv2.putText(frame, f"{predicted_class} ({confidence:.2%})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-time Prediction', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Use the corrected path
#video_path = r'M:/PROJECTS/AI_PROJECT/TEST_VIDEO/TEST_VIDEO.mp4'
video_path = r'M:/PROJECTS/AI_PROJECT/TEST_VIDEO/healthy_video.mp4'
realtime_video_predict(video_path)
