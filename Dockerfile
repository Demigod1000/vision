FROM python:3.9-slim

# Install system dependencies for OpenCV, Tesseract, Tkinter, and PyAudio
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    python3-tk \
    portaudio19-dev \
    alsa-utils \
    pulseaudio \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application code
COPY . .

# Set default environment variables for X11 forwarding and Audio
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# Command to run the application
CMD ["python", "main.py"]
