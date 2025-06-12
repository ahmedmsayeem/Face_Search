import os
import requests
from urllib.parse import urlparse
import uuid
import cv2
from werkzeug.utils import secure_filename
def save_uploaded_file(file, upload_folder):
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    return file_path


def download_image_from_url(url, upload_folder):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None, None, "Failed to download image"

        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type:
            return None, None, "URL does not point to an image"

        ext = content_type.split('/')[-1].split(';')[0]
        if ext not in ['jpeg', 'jpg', 'png', 'bmp']:
            ext = 'jpg'

        parsed = urlparse(url)
        base = os.path.basename(parsed.path)
        if not base:
            base = str(uuid.uuid4())
        filename = f"{os.path.splitext(base)[0]}_{uuid.uuid4().hex[:8]}.{ext}"
        file_path = os.path.join(upload_folder, filename)

        with open(file_path, 'wb') as f:
            f.write(response.content)

        # Extra validation
        if cv2.imread(file_path) is None:
            os.remove(file_path)
            return None, None, "Downloaded file is not a valid image"

        # print(f"[INFO] Downloaded image from URL: {url}")
        print(f"[INFO] Saved to: {file_path}")
        print(f"[INFO] Content-Type: {content_type}")

        return filename, file_path, None

    except Exception as e:
        return None, None, str(e)

