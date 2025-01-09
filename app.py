from flask import Flask, render_template, Response
import cv2
import face_recognition
import pandas as pd
import csv
import time

app = Flask(__name__)

# Load database
database = pd.read_csv('database.csv')

# Load known faces and names
known_face_encodings = []
known_face_names = []
known_face_rolls = []

for i, row in database.iterrows():
    image = face_recognition.load_image_file(row['ImagePath'])
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(row['Name'])
    known_face_rolls.append(row['Roll'])

# Attendance record dictionary
attendance_record = {}

# Function to generate video frames
def generate_frames():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            roll = None

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                roll = known_face_rolls[first_match_index]

                # Mark attendance only if not already marked
                if name not in attendance_record:
                    attendance_record[name] = True
                    with open('attendance.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([name, roll, time.ctime()])
                    print(f"Attendance marked for {name} (Roll: {roll}) at {time.ctime()}")

            # Draw rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Encode the frame for live video feed
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
