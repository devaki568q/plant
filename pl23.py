import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from pymongo import MongoClient
from datetime import datetime
import base64
from io import BytesIO

# Load the model
model = tf.keras.models.load_model('plant.keras')

# Print model summary to verify input shape
model.summary()

# Initialize camera
cap = cv2.VideoCapture(0)  # Use the appropriate camera index or path

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Update with your MongoDB URI
db = client['plant_health_db']  # Database name
collection = db['plant_health']  # Collection name

# Define class labels for various plants
class_labels = [
    'healthy',
    'poe',
    'rust']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL Image for preprocessing
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Resize the image to match the model's expected input dimensions
    image = image.resize((150, 150))  # Adjust size according to model's input shape
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Debug information
    print(f"Image shape: {image_array.shape}")

    # Predict the class
    try:
        predictions = model.predict(image_array)
        class_index = np.argmax(predictions, axis=1)[0]
        label = class_labels[class_index]

        # Display the result on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Convert frame to base64 for MongoDB
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare data for MongoDB
        data = {
            'timestamp': datetime.now(),
            'label': label,
            'frame': img_base64  # Store image as base64 string
        }
        
        # Insert data into MongoDB
        result = collection.insert_one(data)
        print(f"Data inserted with id: {result.inserted_id}")

    except Exception as e:
        print(f"Error during prediction or MongoDB insertion: {e}")

    cv2.imshow('Leaf Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 