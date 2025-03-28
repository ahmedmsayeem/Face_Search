from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import base64
from io import BytesIO
from PIL import Image
import dlib
import face_recognition

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

def save_uploaded_file(file, upload_folder):
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    return file_path

def process_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        return None, "Invalid image file"

    height, width, channels = image.shape

    # Use face_recognition for face detection
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)

    # Convert face locations to OpenCV rectangle format and draw them
    faces = []
    for (top, right, bottom, left) in face_locations:
        faces.append((left, top, right - left, bottom - top))
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)

    # Print the coordinates of the identified faces
    print("Identified faces:", faces)

    # Convert the processed image to a base64 string
    _, buffer = cv2.imencode('.jpg', image)
    image_blob = base64.b64encode(buffer).decode('utf-8')

    stats = {
        "dimensions": f"{width}x{height}",
        "channels": channels,
        "faces": len(faces),
        "image_blob": image_blob
    }
    return stats, None

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    file_path = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])

    stats, error = process_image(file_path)
    if error:
        return error

    stats["filename"] = file.filename
    return render_template('stats.html', stats=stats)

if __name__ == '__main__':
    app.run(debug=True)
