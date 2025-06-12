from flask import Flask, render_template, request, jsonify
import os
import base64

from face_utils import (
    extract_and_store_encodings,
    process_image,
    find_similar_images,
    face_detector,
    face_rec_model,
    shape_predictor,
    extract_and_store_encodings_to_file,
)
from file_utils import save_uploaded_file, download_image_from_url
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    image_files = [
        f for f in os.listdir(app.config['UPLOAD_FOLDER'])
        if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))
    ]
    image_blobs = []
    for image_file in image_files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file)
        with open(file_path, "rb") as img_file:
            blob = base64.b64encode(img_file.read()).decode('utf-8')
            image_blobs.append(blob)
    return render_template('index.html', image_blobs=image_blobs)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    file_path = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
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
        path = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
        extract_and_store_encodings(path, file.filename)
        saved_files.append(file.filename)
    return jsonify({"message": "Files uploaded successfully", "files": saved_files})

@app.route('/find-similar', methods=['POST'])
def find_similar():
    if 'image' not in request.files:
        return "No reference image provided", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected reference image", 400

    ref_path = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
    matches = find_similar_images(ref_path)
    blobs = []
    for fname in matches:
        path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        with open(path, 'rb') as f:
            blobs.append(base64.b64encode(f.read()).decode('utf-8'))
    return render_template('similar.html', similar_image_blobs=blobs)

@app.route('/upload-url', methods=['POST'])
def upload_url():
    data = request.get_json()
    if not data or 'image_url' not in data:
        return jsonify({"error": "Missing image_url"}), 400

    image_url = data['image_url']
    filename, file_path, error = download_image_from_url(image_url, app.config['UPLOAD_FOLDER'])
    if error:
        return jsonify({"error": error}), 400

    # Try to store encodings and check if faces were found
    result = extract_and_store_encodings_to_file(file_path, filename, "url_encodings.json")
    if result == "no_faces":
        return jsonify({"error": "No faces found in the image"}), 400
    if result == "invalid_image":
        return jsonify({"error": "Downloaded file is not a valid image"}), 400

    return jsonify({"message": "Image processed and encodings stored", "filename": filename})

@app.route('/check-url-image', methods=['POST'])
def check_url_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded image temporarily
    temp_path = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])

    # Extract encoding from uploaded image
    from face_utils import get_face_encodings_from_file
    encodings = get_face_encodings_from_file(temp_path)
    if not encodings:
        return jsonify({"error": "No face found in uploaded image"}), 400

    # Load url_encodings.json and compare
    import numpy as np
    import json
    url_encodings_file = "url_encodings.json"
    if not os.path.exists(url_encodings_file):
        return jsonify({"error": "No URL encodings found"}), 404

    with open(url_encodings_file, 'r') as f:
        url_data = json.load(f)

    threshold = 0.6
    matched_filenames = []
    for fname, stored_encs in url_data.items():
        for stored_enc in stored_encs:
            for uploaded_enc in encodings:
                dist = np.linalg.norm(np.array(uploaded_enc) - np.array(stored_enc))
                if dist < threshold:
                    matched_filenames.append(fname)
                    break

    if not matched_filenames:
        return jsonify({"match": False, "urls": []})

    # If you have a mapping from filename to URL, you can return URLs here.
    # For now, just return the filenames.
    return jsonify({"match": True, "filenames": matched_filenames})

@app.route('/check-url-image-by-url', methods=['POST'])
def check_url_image_by_url():
    data = request.get_json()
    if not data or 'image_url' not in data:
        return jsonify({"error": "Missing image_url"}), 400

    image_url = data['image_url']
    filename, file_path, error = download_image_from_url(image_url, app.config['UPLOAD_FOLDER'])
    if error:
        return jsonify({"error": error}), 400

    from face_utils import get_face_encodings_from_file
    encodings = get_face_encodings_from_file(file_path)
    if not encodings:
        return jsonify({"error": "No face found in downloaded image"}), 400

    import numpy as np
    import json
    url_encodings_file = "url_encodings.json"
    if not os.path.exists(url_encodings_file):
        return jsonify({"error": "No URL encodings found"}), 404

    with open(url_encodings_file, 'r') as f:
        url_data = json.load(f)

    threshold = 0.6
    matched_filenames = []
    for fname, stored_encs in url_data.items():
        for stored_enc in stored_encs:
            for uploaded_enc in encodings:
                dist = np.linalg.norm(np.array(uploaded_enc) - np.array(stored_enc))
                if dist < threshold:
                    matched_filenames.append(fname)
                    break

    if not matched_filenames:
        return jsonify({"match": False, "filenames": []})

    return jsonify({"match": True, "filenames": matched_filenames})

if __name__ == '__main__':
    app.run(debug=True)
