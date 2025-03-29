# Use an official Ubuntu image as the base image
FROM ubuntu:20.04

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-venv \
    python3-pip \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Create the models directory and download the required models
RUN mkdir -p /app/models && \
    wget -O /app/models/dlib_face_recognition_resnet_model_v1.dat https://github.com/ageitgey/face_recognition_models/raw/master/face_recognition_models/models/dlib_face_recognition_resnet_model_v1.dat && \
    wget -O /app/models/shape_predictor_68_face_landmarks.dat https://github.com/GuoQuanhao/68_points/raw/master/shape_predictor_68_face_landmarks.dat

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Set the default command to run when the container starts
CMD ["bash"]