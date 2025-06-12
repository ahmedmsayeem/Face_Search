FROM python:3.12.3-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system packages for dlib
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        wget \
        curl \
        git \
        libopenblas-dev \
        liblapack-dev \
        libx11-dev \
        libgtk-3-dev \
        libboost-all-dev \
        && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Install pip and requirements
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install dlib from source

COPY . .

# Download dlib model files
# BAD PRACTICE: Forcing download from blob and resolving raw link
RUN wget https://github.com/ageitgey/face_recognition_models/blob/master/face_recognition_models/models/dlib_face_recognition_resnet_model_v1.dat -O raw.html && \
    grep -o 'https://github.com/[^"]*raw[^"]*dlib_face_recognition_resnet_model_v1.dat' raw.html | head -n 1 | xargs wget -O models/dlib_face_recognition_resnet_model_v1.dat && \
    rm raw.html

# Expose Flask default port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
