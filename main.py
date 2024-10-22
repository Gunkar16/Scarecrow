import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/imageclassifierOld.h5', compile=False)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Failed to open webcam.")
    exit()

# Initialize serial communication with Arduino
# arduino_port = 'COM12'  # Update this with the correct port for your Arduino
# ser = serial.Serial(arduino_port, 9600, timeout=1)
# time.sleep(2)  # Allow time for the connection to establish

# Open a file to write the detected class
file_path = 'detected_class.txt'

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Rotate the frame by 90 degrees clockwise
    frame = cv2.transpose(frame)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Flip the frame vertically
    frame = cv2.flip(frame, 0)
    # Resize the frame to the required dimensions (256x256)
    resize = cv2.resize(frame, (256, 256))

    # Preprocess the frame by normalizing pixel values
    resize = np.expand_dims(resize, axis=0)  # Add batch dimension
    resize = resize / 255.0  # Normalize pixel values

    # Predict the class
    yhat = model.predict(resize)

    # Apply a threshold to the prediction confidence
    threshold = 0.5
    if yhat > threshold:
        predicted_class = 'Nothing'
        # ser.write(b'Mouse\n')  # Send "Mouse" followed by newline character
    elif yhat < 0.1:
        predicted_class = 'Nothing'
    else:
        predicted_class = 'Crow'
        # ser.write(b'Alexa\n')  # Send "Alexa" followed by newline character

    # Write the detected class to the file
    with open(file_path, 'w') as file:
        file.write(predicted_class)
    
    # Display the predicted class on top-left corner of the frame
    cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,0), 2)

    # Display the frame
    cv2.imshow('Image Classfication Model', frame)

    # Set the video window  to always appear on top
    cv2.setWindowProperty('Image Classfication Model', cv2.WND_PROP_TOPMOST, 1)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close serial connection
cap.release()
cv2.destroyAllWindows()
# ser.close()  # Close serial connection
