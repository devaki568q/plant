import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('plant_health_model.keras')

# Print model summary to verify input shape
model.summary()

# Initialize camera
cap = cv2.VideoCapture(0)  # Use the appropriate camera index or path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL Image for preprocessing
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Resize the image to match the model's expected input dimensions
    image = image.resize((148, 148))  # Adjust size according to model's input shape
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Debug information
    print(f"Image shape: {image_array.shape}")

    # Predict the class
    try:
        predictions = model.predict(image_array)
        class_index = np.argmax(predictions, axis=1)[0]

        # Define class labels
        class_labels = ['healthy', 'multiple_diseases', 'rust', 'scab']
        label = class_labels[class_index]

        # Display the result on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    except Exception as e:
        print(f"Error during prediction: {e}")

    cv2.imshow('Leaf Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
