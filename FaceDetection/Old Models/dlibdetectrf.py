import cv2
import numpy as np
import pickle
import dlib

# Load the pre-trained face detector, shape predictor, and face recognition model from dlib
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download from dlib's model repo
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')  # Download from dlib's model repo

# Load the trained Random Forest model
with open('dlibrf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Load the saved face embeddings and names
with open('face_embeddings.pkl', 'rb') as file:
    faces, person_names = pickle.load(file)

# Confidence threshold for "Unknown"
confidence_threshold = 0.6  # Adjust this threshold as needed

# Start the webcam
cap = cv2.VideoCapture(0)

# Function to extract face embedding using dlib
def get_face_embedding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)  # Detect faces
    
    if len(faces) == 0:
        return None

    embeddings = []
    for face in faces:
        landmarks = sp(gray, face)  # Detect landmarks
        face_descriptor = face_rec_model.compute_face_descriptor(image, landmarks)
        embeddings.append(np.array(face_descriptor))
    
    return embeddings

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

    # Detect faces in the frame
    faces_rects = detector(gray)

    for face in faces_rects:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        face_image = frame[y:y+h, x:x+w]

        # Get face embedding
        embeddings = get_face_embedding(frame)

        if embeddings:
            for embedding in embeddings:
                prob = rf_model.predict_proba([embedding])  # Get the probabilities for each class

                # Get the index of the class with the highest probability
                class_idx = np.argmax(prob)

                # Get the confidence score (probability of the predicted class)
                confidence = prob[0][class_idx]

                if confidence < confidence_threshold:
                    label = "Unknown"
                else:
                    label = person_names[class_idx]  # Use the index to retrieve the person's name

                # Draw a bounding box around the face and show the predicted label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} ({confidence:.2f})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow("Real-Time Face Recognition", frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
