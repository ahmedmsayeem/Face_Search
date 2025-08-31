import json
from flask import Flask, render_template, request, jsonify
import os
import base64
import sqlite3


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

UPLOADS_LINK_FOLDER = 'uploads-link'
os.makedirs(UPLOADS_LINK_FOLDER, exist_ok=True)
app.config['UPLOADS_LINK_FOLDER'] = UPLOADS_LINK_FOLDER

conn = sqlite3.connect('face_search.db')
c = conn.cursor()

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

    # Load images from uploads-link folder
    url_image_blobs = []
    url_image_filenames = []
    uploads_link_folder = app.config['UPLOADS_LINK_FOLDER']
    if os.path.exists(uploads_link_folder):
        url_image_files = [
            f for f in os.listdir(uploads_link_folder)
            if os.path.isfile(os.path.join(uploads_link_folder, f))
        ]
        for image_file in url_image_files:
            file_path = os.path.join(uploads_link_folder, image_file)
            with open(file_path, "rb") as img_file:
                blob = base64.b64encode(img_file.read()).decode('utf-8')
                url_image_blobs.append(blob)
                url_image_filenames.append(image_file)

    return render_template('index.html', image_blobs=image_blobs, url_image_blobs=url_image_blobs, url_image_filenames=url_image_filenames)

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
    category = data.get('category')

    print(image_url, category)

    # Download to a temp location first
    filename, temp_path, error = download_image_from_url(image_url, app.config['UPLOAD_FOLDER'])
    if error:
        return jsonify({"error": error}), 400

    # Store encodings and the original URL
    result = extract_and_store_encodings_to_file(temp_path, filename, "url_encodings.json", image_url=image_url, category=category)

    # Move the image to uploads-link folder for display
    try:
        os.remove(temp_path)
    except Exception:
        pass

    if result == "no_faces":
        return jsonify({"error": "No faces found in the image"}), 400
    if result == "invalid_image":
        return jsonify({"error": "Downloaded file is not a valid image"}), 400

    return jsonify({"message": "Image processed and encodings stored", "filename": filename, "image_url": image_url})

@app.route('/check-url-image', methods=['POST'])
def check_url_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    category = request.form.get('category')  # Get category from form-data

    # Save the uploaded image temporarily
    temp_path = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])

    # Extract encoding from uploaded image
    from face_utils import get_face_encodings_from_file
    encodings = get_face_encodings_from_file(temp_path)
    if not encodings:
        return jsonify({"error": "No face found in uploaded image"}), 400

    import numpy as np
    import sqlite3
    threshold = 0.6
    matched_filenames = []
    matched_urls = []

    conn = sqlite3.connect('face_search.db')
    c = conn.cursor()

    # If category is provided, filter by category
    if category:
        c.execute("""
            SELECT images.link, images.encoding, categories.name
            FROM images
            JOIN categories ON images.category_id = categories.id
            WHERE categories.name = ?
        """, (category,))
    else:
        c.execute("SELECT link, encoding FROM images")

    rows = c.fetchall()
    conn.close()

    for row in rows:
        url = row[0]
        encoding_blob = row[1]
        try:
            stored_encs = json.loads(encoding_blob)
        except Exception:
            continue
        for stored_enc in stored_encs:
            for uploaded_enc in encodings:
                dist = np.linalg.norm(np.array(uploaded_enc) - np.array(stored_enc))
                if dist < threshold:
                    matched_filenames.append(url)
                    matched_urls.append(url)
                    break

    if not matched_filenames:
        return jsonify({"match": False, "urls": []})

    return jsonify({"match": True, "filenames": matched_filenames, "urls": matched_urls})


@app.route('/check-image-by-url', methods=['POST'])
def check_image_by_url():
    data = request.get_json()
    if not data or 'image_url' not in data:
        return jsonify({"error": "Missing image_url"}), 400

    image_url = data['image_url']
    category = data.get('category')  # Optional category filter

    filename, file_path, error = download_image_from_url(image_url, app.config['UPLOAD_FOLDER'])
    if error:
        return jsonify({"error": error}), 400

    from face_utils import get_face_encodings_from_file
    encodings = get_face_encodings_from_file(file_path)
    # Remove downloaded file after extraction
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass

    if not encodings:
        return jsonify({"error": "No face found in downloaded image"}), 400

    import numpy as np
    import sqlite3
    threshold = 0.6
    matched_filenames = []
    matched_urls = []

    conn = sqlite3.connect('face_search.db')
    c = conn.cursor()

    # Category filtering, if provided
    if category:
        c.execute("""
            SELECT images.link, images.encoding, categories.name
            FROM images
            JOIN categories ON images.category_id = categories.id
            WHERE categories.name = ?
        """, (category,))
    else:
        c.execute("SELECT link, encoding FROM images")

    rows = c.fetchall()
    conn.close()

    for row in rows:
        url = row[0]
        encoding_blob = row[1]
        try:
            stored_encs = json.loads(encoding_blob)
        except Exception:
            continue
        for stored_enc in stored_encs:
            for uploaded_enc in encodings:
                dist = np.linalg.norm(np.array(uploaded_enc) - np.array(stored_enc))
                print(f"Distance between encodings: {dist}")
                if dist < threshold:
                    matched_filenames.append(url)
                    matched_urls.append(url)
                    break

    if not matched_filenames:
        return jsonify({"match": False, "urls": [], "filenames": []})

    return jsonify({"match": True, "urls": matched_urls})


@app.route('/add-category', methods=['POST'])
def add_category():
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({"error": "Missing category name"}), 400

    category_name = data['name']

    conn = sqlite3.connect('face_search.db')
    c = conn.cursor()
    c.execute("INSERT INTO categories (name) VALUES (?)", (category_name,))
    conn.commit()
    conn.close()

    return jsonify({"message": "Category added successfully"}), 201

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')




