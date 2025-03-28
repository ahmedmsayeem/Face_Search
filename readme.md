# Face Search

This project is a Flask-based web application for detecting faces in an uploaded image, extracting face encodings, and comparing two faces to determine if they match.

## Features
- Detect faces in an uploaded image.
- Extract face encodings using `dlib`.
- Compare two faces to check if they match.
- Simple and user-friendly web interface.

## Prerequisites
- Python 3.7 or higher
- `pip` (Python package manager)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Face_Search
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Download Required Model Files
Download the following `.dat` files and place them in the `models` directory:
- [dlib_face_recognition_resnet_model_v1.dat](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)
- [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

Extract the `.bz2` files and move the `.dat` files into the `models` directory.

### 5. Run the Application
Start the Flask application:
```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`.

### 6. Usage
1. Open the application in your browser.
2. Upload an image with two faces to compare.
3. The application will display the number of detected faces and whether the two faces match.

## Generate `requirements.txt`
To generate a `requirements.txt` file from your virtual environment, run:
```bash
pip freeze > requirements.txt
```

## Project Structure
```
Face_Search/
├── app.py                 # Main application file
├── templates/
│   └── index.html         # HTML template for the web interface
├── uploads/               # Directory for uploaded images
├── models/                # Directory for model files
├── venv/                  # Virtual environment (ignored by .gitignore)
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

## License
This project is licensed under the MIT License.