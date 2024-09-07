import tensorflow as tf
from pymongo import MongoClient
import cv2
import numpy as np
import base64

# Load the model
model = tf.keras.models.load_model('plant_health_model.keras')

# Print the model summary for debugging
model.summary()

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['plant_database']
collection = db['plant']

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for prediction
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Print the shape of the input image
    print(f'Input image shape: {img.shape}')

    try:
        # Predict plant health
        prediction = model.predict(img)
        healthy, multiple_diseases, rust, scab = prediction[0]

        # Print raw prediction values for debugging
        print(f'Raw prediction values: Healthy={healthy}, Multiple Diseases={multiple_diseases}, Rust={rust}, Scab={scab}')

        # Adjusted logic to determine the health status
        if healthy > 0.50:
            health_status = "Healthy"
        else:
            health_status = "Disease"
        
        print(f'Health Status: {health_status}')  # Print the determined health status for debugging

        # Create a mask for debugging purposes
        mask = np.zeros_like(frame)
        cv2.putText(mask, health_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 255, 0) if health_status == "Healthy" else (0, 0, 255), 2)

        # Display the result on the frame
        frame = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)
        cv2.imshow('Plant Health Detector', frame)

        # Store data in MongoDB
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        plant_data = {
            "image": img_base64,
            "status": health_status
        }
        collection.insert_one(plant_data)

    except Exception as e:
        print(f'Error: {e}')

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
