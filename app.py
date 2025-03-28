from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import dlib  # Use dlib for face detection and encoding

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
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces using dlib
    detected_faces = face_detector(rgb_image, 1)
    face_encodings = []  # List to store face signatures

    # Mark detected faces and extract face encodings
    for face in detected_faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Get the landmarks and compute the face encoding
        shape = shape_predictor(rgb_image, face)
        encoding = face_rec_model.compute_face_descriptor(rgb_image, shape)
        face_encodings.append(list(encoding))  # Convert encoding to a list for JSON serialization
    
    face_matched = False
    if len(face_encodings) == 2:
        distance = np.linalg.norm(np.array(face_encodings[0]) - np.array(face_encodings[1]))
        if distance < 0.6:  # Threshold (can vary)
            print("Faces match!")
            face_matched=True
        else:
            print("Faces do not match.")
    # Convert the processed image to a base64 string
    _, buffer = cv2.imencode('.jpg', image)
    image_blob = base64.b64encode(buffer).decode('utf-8')
    # print(face_encodings)

    stats = {
        "dimensions": f"{width}x{height}",
        "channels": channels,
        "faces": len(detected_faces),
        "face_signatures": len(face_encodings),  # Include face signatures in the stats
        "image_blob": image_blob,
        "face_matched": face_matched,
    }
    return stats, None

def find_similar_images(reference_image_path, threshold=0.6):
    """
    Compare a reference image with all images in the uploads folder to find similar images.
    :param reference_image_path: Path to the reference image.
    :param threshold: Distance threshold for face similarity.
    :return: List of similar image filenames.
    """
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        return "Invalid reference image file"

    # Convert reference image to RGB and extract face encodings
    reference_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    detected_faces = face_detector(reference_rgb, 1)
    if len(detected_faces) == 0:
        return "No faces found in the reference image"

    reference_encodings = []
    for face in detected_faces:
        shape = shape_predictor(reference_rgb, face)
        encoding = face_rec_model.compute_face_descriptor(reference_rgb, shape)
        reference_encodings.append(np.array(encoding))

    # Iterate through all images in the uploads folder
    similar_images = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.isfile(file_path):
            continue

        # Process each image
        image = cv2.imread(file_path)
        if image is None:
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detected_faces = face_detector(rgb_image, 1)
        for face in detected_faces:
            shape = shape_predictor(rgb_image, face)
            encoding = face_rec_model.compute_face_descriptor(rgb_image, shape)
            encoding = np.array(encoding)

            # Compare with reference encodings
            for ref_encoding in reference_encodings:
                distance = np.linalg.norm(ref_encoding - encoding)
                if distance < threshold:
                    similar_images.append(filename)
                    break

    return similar_images

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

@app.route('/upload-multiple', methods=['POST'])
def upload_multiple():
    if 'images' not in request.files:
        return "No file part"
    
    files = request.files.getlist('images')
    if not files:
        return "No selected files"

    saved_files = []
    for file in files:
        if file.filename == '':
            continue
        file_path = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
        saved_files.append(file_path)

    return jsonify({"message": "Files uploaded successfully", "files": saved_files})

@app.route('/find-similar', methods=['POST'])
def find_similar():
    if 'image' not in request.files:
        return "No reference image provided"

    file = request.files['image']
    if file.filename == '':
        return "No selected reference image"

    # Save the reference image temporarily
    reference_image_path = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])

    # Find similar images
    similar_images = find_similar_images(reference_image_path)
    return jsonify({"similar_images": similar_images})

if __name__ == '__main__':
    app.run(debug=True)