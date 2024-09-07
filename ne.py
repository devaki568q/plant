import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('plannt_health_modelll.keras')

# Access the camera
cap = cv2.VideoCapture(0)

# Define your class labels (ensure this matches your model)
class_labels = ['healthy', 'rust', 'scab', 'multiple_diseases',
                'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
                'Potato___Early_blight', 'Potato___healthy', 'Potato___diesase',
                'Tomato_Bacterial_spot', 'Tomato_healthy']

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, (150, 150))  # Resize to match your model input size (150, 150)
    img = img.astype('float32') / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict the class
    predictions = model.predict(img)
    class_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][class_index]

    # Debugging: Print the class_index and predictions
    print(f"Class Index: {class_index}, Predictions: {predictions}")

    # Check if class_index is within the range of class_labels
    if class_index < len(class_labels):
        label = f"{class_labels[class_index]}: {confidence*100:.2f}%"
    else:
        label = f"Unknown class index: {class_index}"

    # Display the label on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Plant Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
