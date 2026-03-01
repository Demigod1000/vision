import time
import math
import threading
import numpy as np
import cv2
import customtkinter as ctk
import speech_recognition as sr
import pytesseract
from PIL import Image, ImageTk
from ultralytics import YOLO

import platform

# Tell pytesseract where the executable is depending on the OS
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

class ObjectDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Setup main window
        self.title("AI Object Detection")
        self.geometry("1400x800")
        self.minsize(800, 600)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # Video Frame
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both", padx=5, pady=5)

        # Controls Frame
        self.controls_frame = ctk.CTkFrame(self, width=250)
        self.controls_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ns")
        self.controls_frame.grid_propagate(False)

        # Controls
        self.title_label = ctk.CTkLabel(self.controls_frame, text="Controls", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(pady=20)

        self.start_btn = ctk.CTkButton(self.controls_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(pady=10, padx=20, fill="x")

        self.stop_btn = ctk.CTkButton(self.controls_frame, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_btn.pack(pady=10, padx=20, fill="x")

        self.conf_label = ctk.CTkLabel(self.controls_frame, text="Confidence Threshold: 50%")
        self.conf_label.pack(pady=(20, 5))
        
        self.conf_slider = ctk.CTkSlider(self.controls_frame, from_=10, to=100, command=self.update_conf)
        self.conf_slider.set(50)
        self.conf_slider.pack(pady=5, padx=20, fill="x")

        self.target_label = ctk.CTkLabel(self.controls_frame, text="Target Objects (comma-separated):")
        self.target_label.pack(pady=(20, 5))

        self.target_entry = ctk.CTkEntry(self.controls_frame, placeholder_text="e.g. person, cup, nose")
        self.target_entry.pack(pady=5, padx=20, fill="x")
        self.target_entry.bind("<Return>", self.confirm_targets)

        self.confirm_btn = ctk.CTkButton(self.controls_frame, text="Confirm Target(s)", command=self.confirm_targets)
        self.confirm_btn.pack(pady=5, padx=20, fill="x")

        self.voice_status_label = ctk.CTkLabel(self.controls_frame, text="Voice: Initializing...", text_color="gray")
        self.voice_status_label.pack(pady=5, padx=20, fill="x")
        
        self.voice_toggle = ctk.CTkSwitch(self.controls_frame, text="Listening Mode", command=self.toggle_voice_listening)
        self.voice_toggle.select() # On by default
        self.voice_toggle.pack(pady=5, padx=20, fill="x")
        
        self.learning_toggle = ctk.CTkSwitch(self.controls_frame, text="Learning Mode", command=self.toggle_learning)
        self.learning_toggle.pack(pady=5, padx=20, fill="x")

        # Variables
        self.cap = None
        self.is_running = False
        self.conf_threshold = 0.5
        self.target_classes = []
        self.listen_thread_active = True
        self.voice_listening = True
        self.learning_mode = False
        self.is_paused_for_learning = False
        self.latest_frame = None
        self.ocr_results = []
        
        # Load both models for simultaneous tracking
        # Upgrade general object detection to YOLO-World (Open-Vocabulary Vision-Language Model)
        self.model = YOLO("yolov8s-worldv2.pt") 
        self.pose_model = YOLO("yolov8n-pose.pt") 

        # Start Voice Recognition Thread
        self.voice_thread = threading.Thread(target=self.listen_for_commands, daemon=True)
        self.voice_thread.start()

        # Start OCR Tracking Thread
        self.ocr_thread = threading.Thread(target=self.run_ocr_loop, daemon=True)
        self.ocr_thread.start()

    def update_conf(self, value):
        self.conf_threshold = int(value) / 100.0
        self.conf_label.configure(text=f"Confidence Threshold: {int(value)}%")

    def confirm_targets(self, event=None):
        raw_text = self.target_entry.get().strip().lower()
        if not raw_text:
            self.target_classes = []
        else:
            self.target_classes = [t.strip() for t in raw_text.split(",") if t.strip()]
            
            # For YOLO-World, we can literally pass the target raw strings into the neural network
            # It will dynamically encode them and search for them zero-shot!
            vlm_targets = [t for t in self.target_classes if t not in ["flexing"] and not t.startswith("text")]
            
            if getattr(self, "learning_mode", False) and "object" not in vlm_targets:
                vlm_targets.append("object")
                
            if vlm_targets:
                self.model.set_classes(vlm_targets)
            else:
                self.model.set_classes(["object"])
        
        # Give visual feedback
        self.confirm_btn.configure(text="Confirmed!")
        self.after(1000, lambda: self.confirm_btn.configure(text="Confirm Target(s)"))

    def start_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            
            # Request 1080p Resolution at 60fps
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            
            if not self.cap.isOpened():
                print("Error: Could not open camera.")
                return
            
            self.is_running = True
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.update_frame()

    def stop_camera(self):
        if self.is_running:
            self.is_running = False
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            if self.cap:
                self.cap.release()
            self.video_label.configure(image=None)

    def update_frame(self):
        if getattr(self, "is_paused_for_learning", False):
            self.after(15, self.update_frame)
            return
            
        if self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Mirror the frame horizontally for a more natural webcam experience
                frame = cv2.flip(frame, 1)
                
                # Save a copy for the OCR thread so it doesn't freeze the main UI loop
                self.latest_frame = frame.copy()

                # Perform Object Detection for general objects
                query_conf = 0.2 if getattr(self, "learning_mode", False) else self.conf_threshold
                results = self.model(frame, conf=query_conf, verbose=False, device=0)
                
                # Perform Pose Estimation for body parts
                pose_results = self.pose_model(frame, conf=self.conf_threshold, verbose=False, device=0)
                
                # Annotate Frame with Iron Man style using BOTH sets of results
                annotated_frame, trigger_learning = self.draw_iron_man_hud(frame, results, pose_results, self.model.names, self.target_classes)

                # Convert to PIL Image and ImageTk
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)

                # Resize the visual output dynamically to fit the current UI window
                lbl_w = self.video_label.winfo_width()
                lbl_h = self.video_label.winfo_height()
                
                # Use actual UI size if available, otherwise fallback to 1280x720 to force UI to expand
                if lbl_w > 10 and lbl_h > 10:
                    img_w, img_h = pil_image.size
                    ratio = min(lbl_w / img_w, lbl_h / img_h)
                    new_w, new_h = int(img_w * ratio), int(img_h * ratio)
                else:
                    new_w, new_h = 1280, 720
                    
                imgtk = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(new_w, new_h))
                self.video_label.configure(image=imgtk)
                self.video_label.image = imgtk
                
                if trigger_learning:
                    self.is_paused_for_learning = True
                    self.after(100, self.prompt_learning)

            # Call this function again after 15ms
            self.after(15, self.update_frame)

    def set_target_from_voice(self, target_name):
        # We append voice commands to existing targets if wanted, or just overwrite.
        # Overwriting is simpler for voice for now:
        self.target_entry.delete(0, 'end')
        self.target_entry.insert(0, target_name)
        self.confirm_targets()
        self.voice_status_label.configure(text=f"Voice: Targeted '{target_name}'", text_color="#00ffcc")

    def update_voice_status(self, text, color="gray"):
        self.voice_status_label.configure(text=text, text_color=color)

    def toggle_voice_listening(self):
        self.voice_listening = self.voice_toggle.get()
        if not self.voice_listening:
            self.update_voice_status("Voice: Disabled/Muted", "gray")
        else:
            self.update_voice_status("Voice: Listening...", "#00ffcc")

    def toggle_learning(self):
        self.learning_mode = self.learning_toggle.get()
        self.confirm_targets()

    def prompt_learning(self):
        dialog = ctk.CTkInputDialog(text="Unknown object detected (<50% confidence).\nWhat is it?", title="Interactive Learning")
        new_label = dialog.get_input()
        
        if new_label and new_label.strip():
            new_label = new_label.strip().lower()
            current = self.target_entry.get().strip()
            if current:
                self.target_entry.delete(0, 'end')
                self.target_entry.insert(0, f"{current}, {new_label}")
            else:
                self.target_entry.insert(0, new_label)
            self.confirm_targets()
            
        self.is_paused_for_learning = False
        
    def listen_for_commands(self):
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            if self.voice_listening:
                self.after(0, self.update_voice_status, "Voice: Listening...", "#00ffcc")

        while self.listen_thread_active:
            if not self.voice_listening:
                time.sleep(0.5)
                continue
                
            try:
                with microphone as source:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                command = recognizer.recognize_google(audio).lower()
                print(f"Heard: {command}")

                # Check for keywords
                if "jarvis" in command:
                    # Look for "target X" or "lock onto X" or "find X"
                    target = None
                    if "read text" in command:
                        target = "text"
                    elif "find text" in command:
                        target_word = command.split("find text")[-1].strip()
                        target = "text " + target_word if target_word else "text"
                    elif "target" in command:
                        target = command.split("target")[-1].strip()
                    elif "lock onto" in command:
                        target = command.split("lock onto")[-1].strip()
                    elif "find" in command:
                        target = command.split("find")[-1].strip()

                    # Clean up common punctuation that might be picked up
                    if target:
                        target = target.replace(".", "").replace("a ", "").replace("the ", "").strip()
                        print(f"Parsed Target: {target}")
                        self.after(0, self.set_target_from_voice, target)

            except sr.WaitTimeoutError:
                pass # Just keep looping
            except sr.UnknownValueError:
                pass # Didn't understand, keep looping
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                self.after(0, self.update_voice_status, "Voice: Error getting results", "red")
                time.sleep(2)
            except Exception as e:
                print(f"Voice error: {e}")
                time.sleep(1)

    def run_ocr_loop(self):
        while self.listen_thread_active:
            # Run OCR as long as we have active targets (since the user might want to find words directly)
            if self.latest_frame is not None and self.target_classes:
                try:
                    gray = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2GRAY)
                    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
                    
                    results = []
                    n_boxes = len(data['text'])
                    for i in range(n_boxes):
                        if int(data['conf'][i]) > 30: # 30% confidence filter
                            word = data['text'][i].strip()
                            if word:
                                print(f"OCR Found: '{word}' conf: {data['conf'][i]}")
                                results.append({
                                    'x': data['left'][i],
                                    'y': data['top'][i],
                                    'w': data['width'][i],
                                    'h': data['height'][i],
                                    'text': word
                                })
                    self.ocr_results = results
                except Exception as e:
                    print(f"OCR Error: {e}")
            else:
                self.ocr_results = []
            
            time.sleep(0.5) # Only run OCR ~2 times a second to prevent CPU overload

    def draw_iron_man_hud(self, frame, results, pose_results, class_names, target_classes=[]):
        trigger_learning = False
        # frame is BGR
        h, w = frame.shape[:2]
        
        # Overlay color: Light Blue / Cyan (BGR format)
        hud_color = (255, 204, 0)
        target_color = (0, 0, 255) # Red for targets
        
        # Add a subtle dark blue/cyan tint overlay to make it feel like a HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (40, 10, 0), -1)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        
        # Center of screen
        cx, cy = w // 2, h // 2
        
        # Add some HUD text at the top and bottom
        cv2.putText(frame, "MK. L MARKER ACTIVE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "TARGETING SYSTEM ONLINE", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"RES: {w}x{h}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 1, cv2.LINE_AA)
        # Time for animations
        t = time.time()
        
        # Body part keypoint mapping for YOLOv8-pose
        body_parts = {
            "nose": 0, "face": 0, "head": 0,
            "left eye": 1, "right eye": 2, "eyes": [1, 2],
            "left ear": 3, "right ear": 4, "ears": [3, 4],
            "left shoulder": 5, "right shoulder": 6, "shoulder": [5, 6], "shoulders": [5, 6],
            "left elbow": 7, "right elbow": 8, "elbow": [7, 8], "elbows": [7, 8],
            "left wrist": 9, "right wrist": 10, "wrist": [9, 10], "wrists": [9, 10], "hand": [9, 10], "hands": [9, 10],
            "left hip": 11, "right hip": 12, "hip": [11, 12], "hips": [11, 12],
            "left knee": 13, "right knee": 14, "knee": [13, 14], "knees": [13, 14],
            "left ankle": 15, "right ankle": 16, "ankle": [15, 16], "ankles": [15, 16], "foot": [15, 16], "feet": [15, 16]
        }

        # Check which of the target_classes are known body parts or relational
        targeted_body_parts = [t for t in target_classes if t in body_parts]
        
        # Parse relational targets (e.g. "person holding cell phone")
        relational_targets = []
        general_targets = []
        
        for t in target_classes:
            if t in body_parts or t.startswith("text"):
                continue
            
            # Very basic NLP heuristic for "holding", "with", "near"
            if " holding " in t:
                subj, obj = t.split(" holding ")
                relational_targets.append((subj.strip(), obj.strip(), "holding"))
            elif " with " in t:
                subj, obj = t.split(" with ")
                relational_targets.append((subj.strip(), obj.strip(), "with"))
            elif " near " in t:
                subj, obj = t.split(" near ")
                relational_targets.append((subj.strip(), obj.strip(), "near"))
            else:
                general_targets.append(t)
                
        # If there are no general targets, we should skip running YOLO-World to save resources
        boxes = None
        if general_targets or relational_targets or getattr(self, "learning_mode", False):
            boxes = results[0].boxes if results else None
        
        pose_boxes = pose_results[0].boxes if pose_results else None
        keypoints = pose_results[0].keypoints if pose_results and hasattr(pose_results[0], 'keypoints') else None

        # --- Helper for Overlap ---
        def check_overlap(boxA, boxB):
            # Returns True if there's any intersection
            ax1, ay1, ax2, ay2 = boxA
            bx1, by1, bx2, by2 = boxB
            
            x_left = max(ax1, bx1)
            y_top = max(ay1, by1)
            x_right = min(ax2, bx2)
            y_bottom = min(ay2, by2)
            
            if x_right < x_left or y_bottom < y_top:
                return False
            return True

        # --- Helper for Angle Calculation ---
        def calculate_angle(a, b, c):
            # Calculate the angle between three points (b is the vertex)
            # Returns angle in degrees
            ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
            ang = ang + 360 if ang < 0 else ang
            ang = 360 - ang if ang > 180 else ang
            return ang

        # Pre-process all general boxes for relational lookups
        all_objects = []
        if boxes is not None:
            for box in boxes:
                name = class_names[int(box.cls[0].item())].lower()
                b_x1, b_y1, b_x2, b_y2 = map(int, box.xyxy[0])
                all_objects.append((name, (b_x1, b_y1, b_x2, b_y2), box.conf[0].item()))

        # 1. First Process General Objects
        if boxes is not None:
            for name, coords, conf in all_objects:
                x1, y1, x2, y2 = coords
                
                is_learning_target = False
                if conf < getattr(self, "conf_threshold", 0.5):
                    if getattr(self, "learning_mode", False) and name == "object" and not trigger_learning:
                        trigger_learning = True
                        is_learning_target = True
                    else:
                        continue
                
                # If ANY target is a body part, we skip "person" here because the pose model handles it
                if targeted_body_parts and name == "person":
                    continue

                if is_learning_target:
                    color_to_use = (0, 165, 255) # Orange for learning!
                    status_text = "LEARNING..."
                    thickness = 3
                    length = 30
                else:
                    # Determine if we should lock-on to the whole object
                    is_target = any(gt in name for gt in general_targets)
                    
                    # Check relational targets
                    for r_subj, r_obj, r_verb in relational_targets:
                        if r_subj in name:
                            # This is our subject (e.g., person). Does it overlap with the object (e.g., cell phone)?
                            for o_name, o_coords, o_conf in all_objects:
                                if r_obj in o_name and check_overlap(coords, o_coords):
                                    is_target = True
                                    break

                    if is_target:
                        color_to_use = target_color
                        status_text = "LOCKED"
                        thickness = 3
                        length = 30
                    else:
                        color_to_use = hud_color
                        status_text = "ACQ"
                        thickness = 2
                        length = 20

                # Draw corners for bounding box
                # Top Left
                cv2.line(frame, (x1, y1), (x1 + length, y1), color_to_use, thickness)
                cv2.line(frame, (x1, y1), (x1, y1 + length), color_to_use, thickness)
                
                # Top Right
                cv2.line(frame, (x2, y1), (x2 - length, y1), color_to_use, thickness)
                cv2.line(frame, (x2, y1), (x2, y1 + length), color_to_use, thickness)
                
                # Bottom Left
                cv2.line(frame, (x1, y2), (x1 + length, y2), color_to_use, thickness)
                cv2.line(frame, (x1, y2), (x1, y2 - length), color_to_use, thickness)
                
                # Bottom Right
                cv2.line(frame, (x2, y2), (x2 - length, y2), color_to_use, thickness)
                cv2.line(frame, (x2, y2), (x2, y2 - length), color_to_use, thickness)
                    
                # Always draw the faint brackets/box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_to_use, 1)

                try:
                    cv2.line(frame, (x2, y1), (x2 + 20, y1 - 20), color_to_use, 1)
                    cv2.line(frame, (x2 + 20, y1 - 20), (x2 + 150, y1 - 20), color_to_use, 1)
                    
                    label = f"[{name.upper()}] {status_text} {conf:.2f}"
                    cv2.putText(frame, label, (x2 + 25, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_to_use, 1, cv2.LINE_AA)
                except Exception:
                    pass

        # 2. Then Process Poses (only if someone is looking for a body part or a person)
        if pose_boxes is not None and keypoints is not None:
            for i, box in enumerate(pose_boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()

                for t_part in targeted_body_parts:
                    kps = keypoints.data[i] # [17, 3] tensor: x, y, conf
                    part_idx = body_parts.get(t_part)
                    
                    if part_idx is not None:
                        indices_to_track = [part_idx] if isinstance(part_idx, int) else part_idx
                        
                        for idx in indices_to_track:
                            kx, ky, kconf = kps[idx]
                            if kconf > 0.5: # Only lock if confident about body part
                                kx, ky = int(kx), int(ky)
                                
                                # Draw a target box at the keypoint
                                length = 15
                                cv2.rectangle(frame, (kx - length, ky - length), (kx + length, ky + length), target_color, 2)
                                
                                # Targeting lines
                                cv2.line(frame, (kx + length, ky - length), (kx + length + 15, ky - length - 15), target_color, 1)
                                cv2.line(frame, (kx + length + 15, ky - length - 15), (kx + length + 100, ky - length - 15), target_color, 1)
                                
                                label = f"[{t_part.upper()}] LOCKED {kconf:.2f}"
                                cv2.putText(frame, label, (kx + length + 20, ky - length - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, target_color, 1, cv2.LINE_AA)
                
                # Check for specific actions like "flexing"
                if "flexing" in target_classes:
                    kps = keypoints.data[i] # [17, 3]
                    
                    # Ensure we have decent confidence on both arms (shoulders, elbows, wrists)
                    arms_idx = [5, 6, 7, 8, 9, 10]
                    if all(kps[idx][2] > 0.4 for idx in arms_idx):
                        # Coordinates
                        ls, rs = kps[5][:2], kps[6][:2]
                        le, re = kps[7][:2], kps[8][:2]
                        lw, rw = kps[9][:2], kps[10][:2]
                        
                        # Calculate angles
                        left_angle = calculate_angle(ls, le, lw)
                        right_angle = calculate_angle(rs, re, rw)
                        
                        # Flexing heuristics:
                        # 1. Elbows are bent (angle between 30 and 130 degrees)
                        # 2. Wrists are physically higher than elbows (y-coord is smaller)
                        left_flexing = 30 < left_angle < 130 and lw[1] < le[1]
                        right_flexing = 30 < right_angle < 130 and rw[1] < re[1]
                        
                        if left_flexing and right_flexing:
                            color_to_use = target_color
                            thickness = 3
                            length = 30
                            
                            # Draw lock-on around the whole person
                            cv2.line(frame, (x1, y1), (x1 + length, y1), color_to_use, thickness)
                            cv2.line(frame, (x1, y1), (x1, y1 + length), color_to_use, thickness)
                            cv2.line(frame, (x2, y1), (x2 - length, y1), color_to_use, thickness)
                            cv2.line(frame, (x2, y1), (x2, y1 + length), color_to_use, thickness)
                            cv2.line(frame, (x1, y2), (x1 + length, y2), color_to_use, thickness)
                            cv2.line(frame, (x1, y2), (x1, y2 - length), color_to_use, thickness)
                            cv2.line(frame, (x2, y2), (x2 - length, y2), color_to_use, thickness)
                            cv2.line(frame, (x2, y2), (x2, y2 - length), color_to_use, thickness)
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color_to_use, 1)
                            
                            cv2.line(frame, (x2, y1), (x2 + 20, y1 - 20), color_to_use, 1)
                            cv2.line(frame, (x2 + 20, y1 - 20), (x2 + 150, y1 - 20), color_to_use, 1)
                            
                            label = f"[FLEXING DETECTED] LOCKED"
                            cv2.putText(frame, label, (x2 + 25, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_to_use, 1, cv2.LINE_AA)

        # 3. Process OCR Text Tracker
        # We iterate over all targets. If the user explicitly typed "text", we search for any text.
        # Otherwise, if they typed a specific word (e.g. "exit"), we scan all OCR words for it.
        for target in target_classes:
            if target in body_parts:
                continue
                
            search_word = target
            if search_word.startswith("text"):
                search_word = search_word.replace("text", "").strip()
                
            for res in getattr(self, "ocr_results", []):
                word = res['text']
                wx, wy, ww, wh = res['x'], res['y'], res['w'], res['h']
                
                # Check if this text matches what we are searching for, or if we just want "all text"
                is_target_word = (search_word == "" or search_word in word.lower())
                
                if is_target_word:
                    # Draw a distinct text tracking box
                    cv2.rectangle(frame, (wx, wy), (wx + ww, wy + wh), target_color, 2)
                    
                    # Targeting line for text
                    cv2.line(frame, (wx + ww, wy), (wx + ww + 15, wy - 15), target_color, 1)
                    cv2.line(frame, (wx + ww + 15, wy - 15), (wx + ww + 100, wy - 15), target_color, 1)
                    
                    label = f"[TEXT: {word.upper()}] LOCKED"
                    cv2.putText(frame, label, (wx + ww + 20, wy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, target_color, 1, cv2.LINE_AA)

        return frame, trigger_learning

    def on_closing(self):
        self.listen_thread_active = False
        self.stop_camera()
        self.destroy()

if __name__ == "__main__":
    app = ObjectDetectionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
