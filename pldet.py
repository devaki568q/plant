import tensorflow as tf
from pymongo import MongoClient
import cv2
import numpy as np
import base64

model = tf.keras.models.load_model('my_model.keras')

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['plant_database']
collection = db['plant_data']

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

    # Predict plant health
    prediction = model.predict(img)
    healthy, multiple_diseases, rust, scab = prediction[0]

    # Determine the health status
    health_status = "Healthy" if healthy > 0.5 else "Unhealthy"

    # Display the result on the frame
    cv2.putText(frame, f'Status: {health_status}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Plant Health Detector', frame)

    # Store data in MongoDB
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    plant_data = {
        "image": img_base64,
        "status": health_status
    }
    collection.insert_one(plant_data)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
