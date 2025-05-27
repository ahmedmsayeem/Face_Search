from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import dlib
import json

ENCODINGS_FILE = "encodings.json"

# Initialize dlib's face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
face_rec_model = dlib.face_recognition_model_v1(
    "models/dlib_face_recognition_resnet_model_v1.dat"
)
shape_predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat"
)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    # Get all image filenames from the uploads folder
    image_files = [
        f for f in os.listdir(app.config['UPLOAD_FOLDER'])
        if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))
    ]

    # Convert images to base64 blobs
    image_blobs = []
    for image_file in image_files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file)
        with open(file_path, "rb") as img_file:
            blob = base64.b64encode(img_file.read()).decode('utf-8')
            image_blobs.append(blob)

    return render_template('index.html', image_blobs=image_blobs)


def save_uploaded_file(file):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    return file_path


def extract_and_store_encodings(file_path, filename):
    image = cv2.imread(file_path)
    if image is None:
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_faces = face_detector(rgb_image, 1)
    encodings = []
    for face in detected_faces:
        shape = shape_predictor(rgb_image, face)
        encoding = face_rec_model.compute_face_descriptor(rgb_image, shape)
        encodings.append(list(encoding))

    if not encodings:
        return

    data = {}
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'r') as f:
            data = json.load(f)

    data[filename] = encodings
    with open(ENCODINGS_FILE, 'w') as f:
        json.dump(data, f)


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    file_path = save_uploaded_file(file)
    extract_and_store_encodings(file_path, file.filename)

    stats, error = process_image(file_path)
    if error:
        return error, 400

    stats["filename"] = file.filename
    return render_template('stats.html', stats=stats)


@app.route('/upload-multiple', methods=['POST'])
def upload_multiple():
    if 'images' not in request.files:
        return "No files part", 400
    files = request.files.getlist('images')
    if not files:
        return "No selected files", 400

    saved_files = []
    for file in files:
        if file.filename == '':
            continue
        path = save_uploaded_file(file)
        extract_and_store_encodings(path, file.filename)
        saved_files.append(file.filename)

    return jsonify({"message": "Files uploaded successfully", "files": saved_files})


def process_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        return None, "Invalid image file"

    height, width, channels = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_faces = face_detector(rgb_image, 1)
    face_encodings = []

    for face in detected_faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        shape = shape_predictor(rgb_image, face)
        encoding = face_rec_model.compute_face_descriptor(rgb_image, shape)
        face_encodings.append(list(encoding))

    face_matched = False
    if len(face_encodings) == 2:
        dist = np.linalg.norm(np.array(face_encodings[0]) - np.array(face_encodings[1]))
        face_matched = dist < 0.6

    _, buffer = cv2.imencode('.jpg', image)
    image_blob = base64.b64encode(buffer).decode('utf-8')

    stats = {
        "dimensions": f"{width}x{height}",
        "channels": channels,
        "faces": len(detected_faces),
        "face_signatures": len(face_encodings),
        "image_blob": image_blob,
        "face_matched": face_matched,
    }
    return stats, None


def find_similar_images(reference_image_path, threshold=0.6):
    image = cv2.imread(reference_image_path)
    if image is None:
        return []

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_faces = face_detector(rgb_image, 1)
    if not detected_faces:
        return []

    reference_encodings = []
    for face in detected_faces:
        shape = shape_predictor(rgb_image, face)
        encoding = face_rec_model.compute_face_descriptor(rgb_image, shape)
        reference_encodings.append(np.array(encoding))

    if not os.path.exists(ENCODINGS_FILE):
        return []

    with open(ENCODINGS_FILE, 'r') as f:
        stored_data = json.load(f)

    similar_images = []
    for filename, encs in stored_data.items():
        for enc in encs:
            arr = np.array(enc)
            for ref_enc in reference_encodings:
                if np.linalg.norm(ref_enc - arr) < threshold:
                    similar_images.append(filename)
                    break
    return similar_images


@app.route('/find-similar', methods=['POST'])
def find_similar():
    if 'image' not in request.files:
        return "No reference image provided", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected reference image", 400

    ref_path = save_uploaded_file(file)
    matches = find_similar_images(ref_path)

    blobs = []
    for fname in matches:
        path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        with open(path, 'rb') as f:
            blobs.append(base64.b64encode(f.read()).decode('utf-8'))

    return render_template('similar.html', similar_image_blobs=blobs)

if __name__ == '__main__':
    app.run(debug=True)
