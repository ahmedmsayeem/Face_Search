# Face Search API

A Flask-based REST API for face detection and similarity matching using ChromaDB vector database for efficient face embeddings storage and search.

## Features
- **ChromaDB Integration**: Fast vector similarity search for face embeddings
- **Local & Free**: Runs entirely on your machine, no cloud dependencies
- **Face Recognition**: Uses dlib's state-of-the-art face recognition models
- **Category Support**: Organize face embeddings by categories
- **Web Interface**: Simple UI for testing and visualization
- **Docker Support**: Easy deployment with Docker

## Quick Setup

### Local Development (Recommended)
```bash
# Clone and navigate
git clone <repo-url>
cd Face_Search

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Docker Deployment
```bash
# Build the image
docker build -t face-search .

# Run with Docker Compose (recommended)
docker-compose up -d

# Or run directly
docker run -p 5000:5000 -v $(pwd)/chroma_db:/app/chroma_db face-search
```

### Using the Start Script
```bash
# Make executable and run
chmod +x start.sh
./start.sh
```

## API Access
- **Web UI**: http://localhost:5000
- **Upload Image**: POST /upload-url-image
- **Search Faces**: POST /check-url-image
- **Categories**: GET /get-categories

## Requirements
- Python 3.8+
- Flask
- ChromaDB (vector database)
- dlib (face recognition)

### Option 1: Docker (Recommended)
```bash
git clone https://github.com/ahmedmsayeem/Face_Search.git
cd Face_Search
docker build -t face-search .
docker run -p 5000:5000 face-search
```

### Option 2: Local Installation
```bash
git clone https://github.com/ahmedmsayeem/Face_Search.git
cd Face_Search
pip install -r requirements.txt
python chroma_init.py  # Initialize ChromaDB and download models
python app.py
```

## API Endpoints

### 1. Upload Image by URL
**Endpoint:** `POST /upload-url-image`

Upload and process an image from a URL to extract face embeddings and store in ChromaDB.

**Parameters:**
- `url` (string, required): Link to the image
- `category` (string, optional): Category name to organize images

**Request Example:**
```bash
curl -X POST http://localhost:5000/upload-url-image \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image1.jpg", "category": "event1"}'
```

**Response Example:**
```json
{
  "success": true,
  "message": "Image uploaded successfully",
  "url": "https://example.com/image1.jpg",
  "category": "event1"
}
```

### 2. Check Similar Images
**Endpoint:** `POST /check-url-image`

Find similar faces by comparing an image URL against stored face embeddings using ChromaDB's vector similarity search.

**Parameters:**
- `url` (string, required): Link to the image to compare
- `category` (string, optional): Limit search within specific category

**Request Example:**
```bash
curl -X POST http://localhost:5000/check-url-image \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/test-image.jpg", "category": "event1"}'
```

**Response Example:**
```json
{
  "match": true,
  "urls": [
    "https://example.com/event1/pic1.jpg",
    "https://example.com/event1/pic2.jpg"
  ]
}
```

### 3. Additional Endpoints

#### Get Categories
```bash
GET /get-categories
# Returns: {"categories": ["event1", "event2", ...]}
```

#### Get Database Stats
```bash
GET /stats
# Returns: {"total_face_embeddings": 150, "collection_name": "face_embeddings"}
```

## Technology Stack

- **Backend**: Flask (Python)
- **Vector Database**: ChromaDB (local, persistent)
- **Face Recognition**: dlib with ResNet models
- **Image Processing**: OpenCV
- **Frontend**: HTML/CSS/JavaScript (simple web UI)

## Why ChromaDB?

✅ **Free & Local**: No cloud dependencies or costs  
✅ **Fast Similarity Search**: Optimized for vector operations  
✅ **Easy to Use**: Simple Python API  
✅ **Persistent**: Data survives restarts  
✅ **Scalable**: Handles thousands of face embeddings efficiently  

## Examples

### JavaScript (Frontend)
```javascript
// Upload image
const uploadResponse = await fetch('/upload-url-image', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    url: 'https://example.com/image.jpg',
    category: 'event1'
  })
});

// Check for similar images
const checkResponse = await fetch('/check-url-image', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    url: 'https://example.com/test.jpg',
    category: 'event1'
  })
});
```

### Python (Backend/API)
```python
import requests

# Upload image
upload_data = {
    "url": "https://example.com/image.jpg",
    "category": "event1"
}
response = requests.post('http://localhost:5000/upload-url-image', json=upload_data)

# Check for similar images
check_data = {
    "url": "https://example.com/test.jpg",
    "category": "event1"
}
response = requests.post('http://localhost:5000/check-url-image', json=check_data)
```

## Error Responses
Common error responses include:
- `400 Bad Request`: Missing required parameters
- `400 Bad Request`: No faces found in image
- `400 Bad Request`: Invalid image format or URL
- `500 Internal Server Error`: Processing error

## Project Structure
```
Face_Search/
├── app.py                          # Main Flask application
├── chroma_face_utils.py            # ChromaDB face search utilities
├── chroma_init.py                  # Database initialization
├── file_utils.py                   # File handling utilities
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── templates/                      # Web UI templates
│   └── index.html
├── models/                         # Face recognition models (auto-downloaded)
├── uploads/                        # Temporary image storage
└── chroma_db/                      # ChromaDB persistent storage
```

---

**GitHub:** [ahmedmsayeem/Face_Search](https://github.com/ahmedmsayeem/Face_Search)  
© 2025 Face Search API
