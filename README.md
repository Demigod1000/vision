# AI Object Detection & HUD Tracking System

This project is an advanced, real-time AI object detection application featuring a futuristic "Iron Man" style Heads-Up Display (HUD). It leverages state-of-the-art zero-shot object detection (YOLO-World), pose estimation (YOLO-Pose), Optical Character Recognition (OCR), and voice command integration to create an interactive computer vision experience.

## Features

- **Zero-Shot Object Detection (YOLO-World):** Detect virtually any object by simply typing its name or speaking it, without needing to retrain the model.
- **Pose Estimation & Action Recognition (YOLO-Pose):** Detect specific body parts (e.g., eyes, wrists, shoulders) and recognize basic actions (e.g., "flexing").
- **Voice Control (SpeechRecognition):** Use natural language commands starting with "Jarvis..." to dynamically set targets and toggle listening modes.
- **Optical Character Recognition (Tesseract OCR):** Detect, read, and track specific words or simply any text in the live video feed.
- **Interactive Learning Mode:** Automatically identify objects with low confidence and prompt the user to label them on the fly.
- **Futuristic HUD UI:** Customized bounding boxes, crosshairs, tracking lines, and status labels built with `OpenCV` and a sleek UI built on `customtkinter`.

## Prerequisites

Before running the application, make sure you have the following installed:

1. **Python 3.8+**
2. **Tesseract OCR:** 
   * **Windows:** Download and install Tesseract from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki). 
   * **Note:** The application expects the Tesseract executable to be located at `C:\Program Files\Tesseract-OCR\tesseract.exe`. If you install it elsewhere, please update the path in `main.py` on line 13.

## Installation

1. **Clone or download the repository:**
   Navigate into the project directory: `e:\Projects\Vision\AI`

2. **(Optional but recommended) Create a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install the required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: `pyaudio` might require additional system build tools depending on your Windows setup.*

4. **Model Weights (Auto-Download):** 
   ### Using Docker (Advanced)

Due to hardware requirements like webcams and microphones, running this application in Docker requires passing through X11 GUI and hardware devices.

**1. Window/WSL2 Users:**

To run GUI apps on Windows from a Docker container in WSL2:
1. Run an X Server like [VcXsrv](https://sourceforge.net/projects/vcxsrv/). 
   - **Crucial step:** When launching XLaunch, ensure **"Disable access control"** is checked.
2. In your WSL2 terminal, find your WSL IP address by running:
   ```bash
   export DISPLAY=$(grep -m 1 nameserver /etc/resolv.conf | awk '{print $2}'):0
   ```
3. Ensure device passthrough (`/dev/video0`) is active if using a webcam via `usbipd-win`.
4. Then build and run the image, ensuring you pass the dynamic DISPLAY variable:
   ```bash
   docker-compose up --build
   ```

**2. Linux Users:**
Simply run:
```bash
docker-compose up --build
```

## Usage

Start the application by running:

```bash
python main.py
```

### Controls & Interface

- **Start/Stop Camera:** Toggles the camera feed.
- **Confidence Threshold:** Adjusts how confident the AI needs to be to lock onto an object (10% - 100%).
- **Target Objects:** Enter comma-separated targets manually (e.g., `person, cell phone, text`).
- **Listening Mode:** Toggles the voice command thread.
- **Learning Mode:** When enabled, the app will pause and prompt you to name unknown objects (confidence < 50%).

### Voice Commands

If your microphone is active and the "Listening Mode" is on, you can control the AI by saying phrases like:

- *"Jarvis, lock onto cell phone"*
- *"Jarvis, target person"*
- *"Jarvis, find keys"*
- *"Jarvis, read text"* (Searches for any visible text)
- *"Jarvis, find text exit"* (Specifically highlights the word "exit")

*(You can replace "target", "find", and "lock onto" interchangeably).*

## Dependencies

- `opencv-python`
- `ultralytics`
- `customtkinter`
- `Pillow`
- `SpeechRecognition`
- `pyaudio`
- `pytesseract`
