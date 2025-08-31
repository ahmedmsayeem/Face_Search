import os
import cv2
import base64
import numpy as np
import dlib
import json
import sqlite3




ENCODINGS_FILE = "encodings.json"

# Initialize dlib's face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
face_rec_model = dlib.face_recognition_model_v1(
    "models/dlib_face_recognition_resnet_model_v1.dat"
)
shape_predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat"
)

def extract_and_store_encodings(file_path, filename):
    image = cv2.imread(file_path)
    if image is None:
        return
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_faces = face_detector(rgb_image, 1)
    print(f"Detected {len(detected_faces)} faces in {filename}")
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

def extract_and_store_encodings_to_file(file_path, filename, encodings_file, image_url=None, category=None):
    print(f"Processing {filename} for encoding...")
    image = cv2.imread(file_path)
    print(f"Image shape: {image.shape if image is not None else 'None'}")
    if image is None:
        print("Failed to read image.")
        return "invalid_image"
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_faces = face_detector(rgb_image, 1)
    print(f"Detected {len(detected_faces)} faces")

    encodings = []
    for face in detected_faces:
        shape = shape_predictor(rgb_image, face)
        encoding = face_rec_model.compute_face_descriptor(rgb_image, shape)
        encodings.append(list(encoding))

    if not encodings:
        print("No encodings found.")
        return "no_faces"

    data = {}
    if os.path.exists(encodings_file):
        with open(encodings_file, 'r') as f:
            data = json.load(f)

    # Store both encodings and URL if provided
    entry = {"encodings": encodings}
    if image_url:
        entry["url"] = image_url
    data[filename] = entry

    # Store in SQLite images table
    conn = sqlite3.connect('face_search.db')
    c = conn.cursor()
    category_id = None
    if category:
        c.execute("SELECT id FROM categories WHERE name = ?", (category,))
        row = c.fetchone()
        if row:
            category_id = row[0]
        else:
            # Optionally, create the category if it doesn't exist
            c.execute("INSERT INTO categories (name) VALUES (?)", (category,))
            category_id = c.lastrowid

    c.execute(
        "INSERT INTO images (link, category_id, encoding) VALUES (?, ?, ?)",
        (image_url, category_id, json.dumps(encodings))
    )
    conn.commit()
    conn.close()

    with open(encodings_file, 'w') as f:
        json.dump(data, f)

    print("Encoding successfully stored.")
    return "success"

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

def get_face_encodings_from_file(file_path):
    image = cv2.imread(file_path)
    if image is None:
        return []
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_faces = face_detector(rgb_image, 1)
    encodings = []
    for face in detected_faces:
        shape = shape_predictor(rgb_image, face)
        encoding = face_rec_model.compute_face_descriptor(rgb_image, shape)
        encodings.append(list(encoding))
    return encodings
