import cv2
import numpy as np
import pickle
import dlib
import concurrent.futures

def get_face_embedding(gray, image):
    faces = detector(gray)
    
    if len(faces) == 0:
        return None

    embeddings = []
    for face in faces:
        landmarks = sp(gray, face)
        face_descriptor = face_rec_model.compute_face_descriptor(image, landmarks)
        embeddings.append(np.array(face_descriptor))
    
    return embeddings

# Load the pre-trained face detector, shape predictor, and face recognition model from dlib
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Load the BallTree model and PCA model
with open('ball_tree.pkl', 'rb') as f:
    ball_tree = pickle.load(f)

with open('pca_model2.pkl', 'rb') as f:
    pca = pickle.load(f)

# Load the saved face embeddings and names
with open('face_embeddings2.pkl', 'rb') as file:
    faces, person_names = pickle.load(file)

# Confidence threshold for "Unknown"
confidence_threshold = 0.8

# Start the webcam
cap = cv2.VideoCapture(0)

def process_face(face_image, gray, frame):
    embeddings = get_face_embedding(gray, frame)
    
    if embeddings:
        embedding = embeddings[0]  # Assuming one face per frame
        embedding_pca = pca.transform([embedding])  # Apply PCA for dimensionality reduction
        distances, indices = ball_tree.query(embedding_pca)  # Find the closest match

        closest_person_name = person_names[indices[0][0]]
        confidence = 1 / (distances[0][0] + 1e-5)

        if confidence < confidence_threshold:
            label = "Unknown"
        else:
            label = closest_person_name

        return label
    return None

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rects = detector(gray)

    # Use multithreading to process faces in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for face in faces_rects:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_image = frame[y:y+h, x:x+w]

            # Process each face in parallel
            futures.append(executor.submit(process_face, face_image, gray, frame))

        # Get results from all the futures
        results = [future.result() for future in futures]

    # Draw bounding boxes and labels
    for i, face in enumerate(faces_rects):
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        label = results[i] if results[i] else "Unknown"
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Real-Time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
