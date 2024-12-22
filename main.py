import cv2
import face_recognition
import numpy as np
import pandas as pd
import os

# Load the database
database = pd.read_csv("database.csv")

# Load known faces and their encodings
known_face_encodings = []
known_face_names = []

for index, row in database.iterrows():
    image_path = row['ImagePath']
    name = row['Name']
    roll = row['Roll']

    # Load image and encode
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    
    known_face_encodings.append(encoding)
    known_face_names.append(f"{name} ({roll})")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

print("Press 'q' to quit.")
while True:
    # Capture a single frame
    ret, frame = video_capture.read()
    
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

    # Detect faces and encode them
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the detected face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        # Find the best match
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"

        # Display the results
        top, right, bottom, left = [v * 4 for v in face_location]  # Scale back up
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Face Recognition', frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
