import cv2
import numpy as np
import dlib
import pickle

# Load the pre-trained face detector and face recognition model from dlib
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the BallTree model and PCA model
with open('ball_tree.pkl', 'rb') as f:
    ball_tree = pickle.load(f)
with open('pca_model2.pkl', 'rb') as f:
    pca = pickle.load(f)

# Load the saved face embeddings and names
with open('face_embeddings2.pkl', 'rb') as file:
    faces, person_names = pickle.load(file)

confidence_threshold = 0.8
cap = cv2.VideoCapture(0)

def get_face_embedding(gray, image, face):
    landmarks = sp(gray, face)
    face_descriptor = face_rec_model.compute_face_descriptor(image, landmarks)
    return np.array(face_descriptor)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use OpenCV Haar Cascade for face detection
    faces_rects = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(50, 50))
    
    if len(faces_rects) > 0:
        # Convert the detected bounding boxes from OpenCV format (x, y, w, h) to dlib format (left, top, right, bottom)
        dlib_faces = [dlib.rectangle(x, y, x + w, y + h) for (x, y, w, h) in faces_rects]

        # Find the largest face by area (width * height)
        largest_face = max(dlib_faces, key=lambda face: face.width() * face.height())
        largest_face_rect = faces_rects[dlib_faces.index(largest_face)]
        
        # Get face embedding
        embedding = get_face_embedding(gray, frame, largest_face)
        
        # Apply PCA for dimensionality reduction
        embedding_pca = pca.transform([embedding])
        
        # Query the BallTree to find the closest match
        distances, indices = ball_tree.query(embedding_pca)
        
        closest_person_name = person_names[indices[0][0]]
        confidence = 1 / (distances[0][0] + 1e-5)

        if confidence < confidence_threshold:
            label = "Unknown"
        else:
            label = closest_person_name

        # Draw the bounding box and label for the largest face
        x, y, w, h = largest_face_rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Real-Time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
