import json
from flask import Flask, render_template, request, jsonify
import os
import base64
import uuid
import tempfile

from chroma_face_utils import ChromaFaceSearch
from file_utils import download_image_from_url

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize ChromaDB face search
face_search = ChromaFaceSearch()

# Create uploads-link folder for web UI
UPLOADS_LINK_FOLDER = 'uploads-link'
os.makedirs(UPLOADS_LINK_FOLDER, exist_ok=True)
app.config['UPLOADS_LINK_FOLDER'] = UPLOADS_LINK_FOLDER

@app.route('/')
def home():
    # Show local uploaded images (for demo purposes)
    image_files = [
        f for f in os.listdir(app.config['UPLOAD_FOLDER'])
        if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))
    ] if os.path.exists(app.config['UPLOAD_FOLDER']) else []

    image_blobs = []
    for image_file in image_files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file)
        try:
            with open(file_path, "rb") as img_file:
                blob = base64.b64encode(img_file.read()).decode('utf-8')
                image_blobs.append(blob)
        except:
            continue

    # Get ChromaDB stats for display
    stats = face_search.get_stats()
    categories = face_search.list_categories()

    return render_template('index.html', 
                         image_blobs=image_blobs, 
                         stats=stats,
                         categories=categories,
                         url_image_blobs=[],  # Keep for template compatibility
                         url_image_filenames=[])

@app.route('/upload-url-image', methods=['POST'])
def upload_url_image():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "Missing url parameter"}), 400

    image_url = data['url']
    category = data.get('category')

    # Download to a temp location first
    filename, temp_path, error = download_image_from_url(image_url, app.config['UPLOAD_FOLDER'])
    if error:
        return jsonify({"error": error}), 400

    # Add to ChromaDB
    success = face_search.add_image(image_url, temp_path, category)

    # Remove the temporary downloaded file
    try:
        os.remove(temp_path)
    except Exception:
        pass

    if not success:
        return jsonify({"error": "No faces found in the image"}), 400

    return jsonify({
        "success": True,
        "message": "Image uploaded successfully",
        "url": image_url,
        "category": category
    })

@app.route('/check-url-image', methods=['POST'])
def check_url_image():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "Missing url parameter"}), 400

    image_url = data['url']
    category = data.get('category')  # Optional category filter

    filename, file_path, error = download_image_from_url(image_url, app.config['UPLOAD_FOLDER'])
    if error:
        return jsonify({"error": error}), 400

    # Search for similar faces using ChromaDB
    similar_faces = face_search.search_similar_faces(file_path, category=category)
    
    # Remove downloaded file after extraction
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass

    if not similar_faces:
        return jsonify({"match": False, "urls": []})

    # Extract URLs from results
    matched_urls = [face['url'] for face in similar_faces]

    return jsonify({"match": True, "urls": matched_urls})

@app.route('/add-category', methods=['POST'])
def add_category():
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({"error": "Missing category name"}), 400

    category_name = data['name'].strip()
    if not category_name:
        return jsonify({"error": "Category name cannot be empty"}), 400

    # For ChromaDB, categories are added automatically when images are uploaded
    # This endpoint validates the category name and confirms readiness
    return jsonify({"message": f"Category '{category_name}' ready for use"}), 201

@app.route('/get-categories', methods=['GET'])
def get_categories():
    categories = face_search.list_categories()
    return jsonify({"categories": categories})

@app.route('/get-categories-images', methods=['GET'])
def get_categories_images():
    """Get all categories with their images (UI compatibility)"""
    if not face_search:
        return jsonify({"categories": []})
    
    try:
        # Get all categories
        categories = face_search.list_categories()
        
        result = []
        for category in categories:
            # Get images for this category from ChromaDB
            category_images = face_search.get_images_by_category(category)
            
            images_list = []
            for image in category_images:
                images_list.append({
                    "id": image.get('id', ''),
                    "link": image.get('url', ''),
                    "url": image.get('url', '')
                })
            
            result.append({
                "id": len(result) + 1,  # Simple ID for UI
                "name": category,
                "images": images_list
            })
        
        return jsonify({"categories": result})
        
    except Exception as e:
        print(f"Error getting categories and images: {e}")
        return jsonify({"categories": []})

@app.route('/upload-multiple', methods=['POST'])
def upload_multiple():
    """Upload multiple files (UI compatibility)"""
    if not face_search:
        return jsonify({"error": "Face search service not available"}), 503
        
    if 'images' not in request.files:
        return jsonify({"error": "No files part"}), 400
    
    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "No selected files"}), 400

    category = request.form.get('category', 'uncategorized')
    
    saved_files = []
    failed_files = []
    
    for file in files:
        if file.filename == '':
            continue
            
        try:
            # Create temporary file
            temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            file.save(temp_path)
            
            # Create a fake URL for the file (since ChromaDB expects URLs)
            fake_url = f"http://localhost:5000/uploads/{temp_filename}"
            
            # Add to ChromaDB
            success = face_search.add_image(fake_url, temp_path, category)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            if success:
                saved_files.append(file.filename)
            else:
                failed_files.append(file.filename)
                
        except Exception as e:
            failed_files.append(f"{file.filename} (error: {str(e)})")
    
    if saved_files:
        message = f"Successfully uploaded: {', '.join(saved_files)}"
        if failed_files:
            message += f". Failed: {', '.join(failed_files)}"
        return jsonify({"message": message, "files": saved_files})
    else:
        return jsonify({"error": f"All uploads failed. {', '.join(failed_files)}"}), 400

@app.route('/stats', methods=['GET'])
def get_stats():
    stats = face_search.get_stats()
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')




