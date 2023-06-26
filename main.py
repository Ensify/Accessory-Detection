import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('glass_model.h5')

# Open a video capture object for the camera
cap = cv2.VideoCapture(0)

# Set up the OpenCV window
cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera', 640, 480)
print("Program running")
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the frame to match the input size expected by the model
    frame_resized = cv2.resize(frame_rgb, (128, 128))

    # Convert the frame to a numpy array
    frame_array = np.array(frame_resized)

    # Preprocess the frame (normalize, expand dimensions if needed, etc.)
    # ...

    # Make a prediction using the model
    prediction = model.predict(np.expand_dims(frame_array, axis=0))
    prediction = prediction.squeeze()
    prediction = "There is Glass" if prediction >= 0.49 else "No Glass"
    # Get the predicted class or any other required information
    # ...

    # Display the frame with the prediction
    cv2.putText(frame, f'{prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
