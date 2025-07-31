import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import subprocess
import os
import torchaudio
import torchaudio.transforms as T
from backend.models import AnimalClassifier, EmotionClassifier, EmotionClassifierDirect
# Fixed import - remove the non-existent function
from firebase_backend.firebase_utils import save_to_firebase

class CombinedAnimalEmotionDetector:
    def __init__(self):
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models and classes
        self.load_animal_model()
        self.load_emotion_models()
        
        # Setup preprocessing
        self.setup_transforms()
        
    def load_animal_model(self):
        """Load the animal classification model"""
        model_path = "backend/model_animal_best.pth"
        self.animal_model = AnimalClassifier().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.animal_model.load_state_dict(checkpoint['model_state_dict'])
        self.animal_model.eval()
        self.animal_classes = checkpoint.get("classes", ["cat", "dog"])
        
    def load_emotion_models(self):
        """Load both audio and vision emotion models"""
        # Load emotion classes
        with open("backend/emotion_classes_audio.txt", "r") as f:
            self.audio_emotion_classes = [line.strip() for line in f.readlines()]
        with open("backend/emotion_classes.txt", "r") as f:
            self.vision_emotion_classes = [line.strip() for line in f.readlines()]
        
        # Load audio emotion model
        self.audio_emotion_model = EmotionClassifierDirect(
            num_classes=len(self.audio_emotion_classes)
        ).to(self.device)
        self.audio_emotion_model.load_state_dict(
            torch.load("backend/audio_emotion.pth", map_location=self.device)
        )
        self.audio_emotion_model.eval()
        
        # Load vision emotion model
        self.vision_emotion_model = EmotionClassifier(
            num_classes=len(self.vision_emotion_classes)
        ).to(self.device)
        self.vision_emotion_model.load_state_dict(
            torch.load("backend/model_emotion.pth", map_location=self.device)
        )
        self.vision_emotion_model.eval()
        
    def setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.animal_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def predict_animal_from_image(self, image):
        """Predict animal from PIL image"""
        input_tensor = self.animal_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.animal_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
            
        return self.animal_classes[pred], confidence
    
    def predict_emotion_from_image(self, image):
        """Predict emotion from PIL image"""
        img_resized = image.resize((224, 224))
        tensor = torch.tensor(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.vision_emotion_model(tensor)
            prob = torch.softmax(output, dim=1)[0]
            idx = prob.argmax().item()
            label = self.vision_emotion_classes[idx]
            conf = float(prob[idx])
            
        return label, conf
    
    def predict_emotion_from_audio(self, audio_path):
        """Predict emotion from audio file"""
        try:
            # Convert mp3 to wav if needed
            if audio_path.endswith(".mp3"):
                wav_path = audio_path.replace(".mp3", ".wav")
                subprocess.run(["ffmpeg", "-i", audio_path, wav_path],
                             check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                audio_path = wav_path
            
            # Load and process audio
            waveform, sr = torchaudio.load(audio_path)
            mel_spec = T.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=512, n_mels=128)(waveform)
            log_mel = torch.log(mel_spec + 1e-9)
            
            # Pad or truncate to fixed length
            if log_mel.shape[-1] < 300:
                log_mel = torch.nn.functional.pad(log_mel, (0, 300 - log_mel.shape[-1]))
            else:
                log_mel = log_mel[:, :, :300]
            
            log_mel = log_mel.expand(3, -1, -1).unsqueeze(0).to(self.device)
            
            with torch.no_data():
                output = self.audio_emotion_model(log_mel)
                prob = torch.softmax(output, dim=1)[0]
                idx = prob.argmax().item()
                label = self.audio_emotion_classes[idx]
                conf = float(prob[idx])
                
            return label, conf
            
        except Exception as e:
            print(f"Audio emotion prediction failed: {e}")
            return None, 0.0
    
    def extract_first_frame(self, video_path):
        """Extract first frame from video"""
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()
        
        if not success:
            raise ValueError("Could not read frame from video.")
        
        # Convert BGR to RGB and return PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    
    def extract_audio_from_video(self, video_path):
        """Extract audio from video file"""
        audio_path = video_path.replace(".mp4", "_temp_audio.wav")
        
        try:
            subprocess.run([
                "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1", audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return audio_path
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            return None
    
    def predict_from_image(self, image_path):
        """Combined prediction from image"""
        image = Image.open(image_path).convert("RGB")
        
        # Predict animal
        animal, animal_conf = self.predict_animal_from_image(image)
        
        # Predict emotion
        emotion, emotion_conf = self.predict_emotion_from_image(image)
        
        return {
            "animal": animal,
            "animal_confidence": animal_conf,
            "emotion": emotion,
            "emotion_confidence": emotion_conf,
            "source": "image"
        }
    
    def predict_from_video(self, video_path):
        """Combined prediction from video"""
        # Extract first frame for animal detection
        frame = self.extract_first_frame(video_path)
        animal, animal_conf = self.predict_animal_from_image(frame)
        
        # Extract audio for emotion detection
        audio_path = self.extract_audio_from_video(video_path)
        audio_emotion, audio_conf = None, 0.0
        
        if audio_path and os.path.exists(audio_path):
            audio_emotion, audio_conf = self.predict_emotion_from_audio(audio_path)
            # Clean up temporary audio file after use
            try:
                os.remove(audio_path)
            except:
                pass
        
        # Get visual emotion from multiple frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > 0:
            frame_idxs = np.linspace(0, total_frames - 1, num=5, dtype=int)
            probs_list = []
            
            for idx in frame_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_img = Image.fromarray(frame_rgb)
                _, conf = self.predict_emotion_from_image(frame_img)
                
                # Get probabilities for averaging
                img_resized = frame_img.resize((224, 224))
                tensor = torch.tensor(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
                tensor = tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.vision_emotion_model(tensor)
                    prob = torch.softmax(output, dim=1)[0]
                    probs_list.append(prob.cpu().numpy())
            
            cap.release()
            
            if probs_list:
                avg_probs = np.mean(probs_list, axis=0)
                vision_idx = int(np.argmax(avg_probs))
                vision_emotion = self.vision_emotion_classes[vision_idx]
                vision_conf = float(avg_probs[vision_idx])
            else:
                vision_emotion, vision_conf = None, 0.0
        else:
            vision_emotion, vision_conf = None, 0.0
        
        # Choose best emotion prediction
        if audio_conf >= vision_conf and audio_emotion:
            final_emotion = audio_emotion
            final_emotion_conf = audio_conf
            emotion_source = "audio"
        elif vision_emotion:
            final_emotion = vision_emotion
            final_emotion_conf = vision_conf
            emotion_source = "visual"
        else:
            final_emotion = "unknown"
            final_emotion_conf = 0.0
            emotion_source = "none"
        
        return {
            "animal": animal,
            "animal_confidence": animal_conf,
            "emotion": final_emotion,
            "emotion_confidence": final_emotion_conf,
            "source": "video",
            "emotion_source": emotion_source,
            "audio_emotion": {"label": audio_emotion, "confidence": audio_conf},
            "visual_emotion": {"label": vision_emotion, "confidence": vision_conf}
        }

class ModernAnimalEmotionGUI:
    def __init__(self):
        self.detector = CombinedAnimalEmotionDetector()
        # Set default user email - no popup required
        self.user_email = "user@example.com"  # Default email for Firebase logging
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the modern GUI with animal theme"""
        self.root = tk.Tk()
        self.root.title("🐾 PetMood Analyzer - AI Animal Emotion Detection")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f0f8ff')
        
        # Configure style
        self.setup_styles()
        
        # Create main container
        self.create_main_layout()
        
    def setup_styles(self):
        """Setup modern styling"""
        style = ttk.Style()
        
        # Configure colors
        self.colors = {
            'primary': '#4a90e2',
            'secondary': '#7fb069', 
            'accent': '#ff6b6b',
            'background': '#f0f8ff',
            'card_bg': '#ffffff',
            'text_dark': '#2c3e50',
            'text_light': '#7f8c8d',
            'success': '#27ae60',
            'warning': '#f39c12',
            'animal_color': '#e74c3c',
            'emotion_color': '#9b59b6'
        }
        
        # Configure ttk styles
        style.configure('Title.TLabel', font=('Segoe UI', 24, 'bold'), 
                       foreground=self.colors['primary'], background=self.colors['background'])
        style.configure('Subtitle.TLabel', font=('Segoe UI', 12), 
                       foreground=self.colors['text_light'], background=self.colors['background'])
        style.configure('Card.TFrame', background=self.colors['card_bg'], relief='raised', borderwidth=1)
        style.configure('Heading.TLabel', font=('Segoe UI', 14, 'bold'), 
                       foreground=self.colors['text_dark'], background=self.colors['card_bg'])
        
    def create_main_layout(self):
        """Create the main layout with modern design"""
        # Header section
        self.create_header()
        
        # Main content area
        main_container = tk.Frame(self.root, bg=self.colors['background'])
        main_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Upload section
        self.create_upload_section(main_container)
        
        # Content area with preview and results
        content_frame = tk.Frame(main_container, bg=self.colors['background'])
        content_frame.pack(fill='both', expand=True, pady=20)
        
        # Left panel - Preview
        self.create_preview_panel(content_frame)
        
        # Right panel - Results
        self.create_results_panel(content_frame)
        
        # Status bar
        self.create_status_bar()
        
    def create_header(self):
        """Create attractive header with gradient-like effect"""
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=120)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        # Title with emoji
        title_label = tk.Label(header_frame, text="🐾 PetMood Analyzer", 
                              font=('Segoe UI', 28, 'bold'), 
                              fg='white', bg=self.colors['primary'])
        title_label.pack(pady=(20, 5))
        
        # Subtitle
        subtitle_label = tk.Label(header_frame, 
                                 text="AI-Powered Animal Recognition & Emotion Detection", 
                                 font=('Segoe UI', 14), 
                                 fg='#e3f2fd', bg=self.colors['primary'])
        subtitle_label.pack()
        
    def create_upload_section(self, parent):
        """Create modern upload section"""
        upload_frame = tk.Frame(parent, bg=self.colors['background'])
        upload_frame.pack(fill='x', pady=20)
        
        # Upload card
        upload_card = tk.Frame(upload_frame, bg=self.colors['card_bg'], 
                              relief='raised', bd=2, padx=30, pady=20)
        upload_card.pack()
        
        # Upload icon and text
        upload_text = tk.Label(upload_card, text="📁 Choose Your Pet's Photo or Video", 
                              font=('Segoe UI', 16, 'bold'), 
                              fg=self.colors['primary'], bg=self.colors['card_bg'])
        upload_text.pack(pady=(0, 10))
        
        # Upload button with modern styling
        self.upload_btn = tk.Button(upload_card, text="🔍 Select Image or Video", 
                                   command=self.upload_file,
                                   font=('Segoe UI', 12, 'bold'),
                                   bg=self.colors['primary'], fg='white',
                                   relief='flat', padx=30, pady=12,
                                   cursor='hand2', activebackground='#357abd')
        self.upload_btn.pack()
        
        # Supported formats
        formats_label = tk.Label(upload_card, 
                                text="Supports: JPG, PNG, MP4, AVI, MOV", 
                                font=('Segoe UI', 10), 
                                fg=self.colors['text_light'], bg=self.colors['card_bg'])
        formats_label.pack(pady=(10, 0))
        
    def create_preview_panel(self, parent):
        """Create modern preview panel"""
        preview_container = tk.Frame(parent, bg=self.colors['background'])
        preview_container.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Preview card
        preview_card = tk.Frame(preview_container, bg=self.colors['card_bg'], 
                               relief='raised', bd=2)
        preview_card.pack(fill='both', expand=True)
        
        # Preview header
        preview_header = tk.Frame(preview_card, bg=self.colors['secondary'], height=50)
        preview_header.pack(fill='x')
        preview_header.pack_propagate(False)
        
        preview_title = tk.Label(preview_header, text="📷 Media Preview", 
                                font=('Segoe UI', 14, 'bold'), 
                                fg='white', bg=self.colors['secondary'])
        preview_title.pack(pady=15)
        
        # Preview content
        preview_content = tk.Frame(preview_card, bg=self.colors['card_bg'])
        preview_content.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Preview display area with proper sizing
        preview_area = tk.Frame(preview_content, bg='#f8f9fa', relief='groove', bd=2)
        preview_area.pack(pady=20, padx=20, fill='both', expand=True)
        
        self.preview_label = tk.Label(preview_area, 
                                     text="🎯 Upload an image or video\nto see preview here", 
                                     font=('Segoe UI', 12), 
                                     fg=self.colors['text_light'], 
                                     bg='#f8f9fa',
                                     compound='center')
        self.preview_label.pack(expand=True)
        
        # File info
        self.file_info_var = tk.StringVar()
        self.file_info_label = tk.Label(preview_content, textvariable=self.file_info_var,
                                       font=('Segoe UI', 10), 
                                       fg=self.colors['text_light'], 
                                       bg=self.colors['card_bg'])
        self.file_info_label.pack()
        
    def create_results_panel(self, parent):
        """Create modern results panel"""
        results_container = tk.Frame(parent, bg=self.colors['background'])
        results_container.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Animal results card
        self.create_animal_card(results_container)
        
        # Emotion results card
        self.create_emotion_card(results_container)
        
        # Summary card
        self.create_summary_card(results_container)
        
    def create_animal_card(self, parent):
        """Create animal detection results card"""
        animal_card = tk.Frame(parent, bg=self.colors['card_bg'], 
                              relief='raised', bd=2)
        animal_card.pack(fill='x', pady=(0, 15))
        
        # Animal header
        animal_header = tk.Frame(animal_card, bg=self.colors['animal_color'], height=50)
        animal_header.pack(fill='x')
        animal_header.pack_propagate(False)
        
        animal_title = tk.Label(animal_header, text="🐕 Animal Detection", 
                               font=('Segoe UI', 14, 'bold'), 
                               fg='white', bg=self.colors['animal_color'])
        animal_title.pack(pady=15)
        
        # Animal content
        animal_content = tk.Frame(animal_card, bg=self.colors['card_bg'])
        animal_content.pack(fill='x', padx=20, pady=20)
        
        # Animal result
        animal_result_frame = tk.Frame(animal_content, bg=self.colors['card_bg'])
        animal_result_frame.pack(fill='x')
        
        tk.Label(animal_result_frame, text="Detected:", 
                font=('Segoe UI', 11), fg=self.colors['text_dark'], 
                bg=self.colors['card_bg']).pack(side='left')
        
        self.animal_var = tk.StringVar(value="Waiting for upload...")
        self.animal_label = tk.Label(animal_result_frame, textvariable=self.animal_var,
                                    font=('Segoe UI', 14, 'bold'), 
                                    fg=self.colors['animal_color'], 
                                    bg=self.colors['card_bg'])
        self.animal_label.pack(side='right')
        
        # Animal confidence
        animal_conf_frame = tk.Frame(animal_content, bg=self.colors['card_bg'])
        animal_conf_frame.pack(fill='x', pady=(10, 0))
        
        tk.Label(animal_conf_frame, text="Confidence:", 
                font=('Segoe UI', 11), fg=self.colors['text_dark'], 
                bg=self.colors['card_bg']).pack(side='left')
        
        self.animal_conf_var = tk.StringVar(value="-")
        self.animal_conf_label = tk.Label(animal_conf_frame, textvariable=self.animal_conf_var,
                                         font=('Segoe UI', 12, 'bold'), 
                                         fg=self.colors['success'], 
                                         bg=self.colors['card_bg'])
        self.animal_conf_label.pack(side='right')
        
    def create_emotion_card(self, parent):
        """Create emotion detection results card"""
        emotion_card = tk.Frame(parent, bg=self.colors['card_bg'], 
                               relief='raised', bd=2)
        emotion_card.pack(fill='x', pady=(0, 15))
        
        # Emotion header
        emotion_header = tk.Frame(emotion_card, bg=self.colors['emotion_color'], height=50)
        emotion_header.pack(fill='x')
        emotion_header.pack_propagate(False)
        
        emotion_title = tk.Label(emotion_header, text="😊 Emotion Analysis", 
                                font=('Segoe UI', 14, 'bold'), 
                                fg='white', bg=self.colors['emotion_color'])
        emotion_title.pack(pady=15)
        
        # Emotion content
        emotion_content = tk.Frame(emotion_card, bg=self.colors['card_bg'])
        emotion_content.pack(fill='x', padx=20, pady=20)
        
        # Emotion result
        emotion_result_frame = tk.Frame(emotion_content, bg=self.colors['card_bg'])
        emotion_result_frame.pack(fill='x')
        
        tk.Label(emotion_result_frame, text="Mood:", 
                font=('Segoe UI', 11), fg=self.colors['text_dark'], 
                bg=self.colors['card_bg']).pack(side='left')
        
        self.emotion_var = tk.StringVar(value="Analyzing...")
        self.emotion_label = tk.Label(emotion_result_frame, textvariable=self.emotion_var,
                                     font=('Segoe UI', 14, 'bold'), 
                                     fg=self.colors['emotion_color'], 
                                     bg=self.colors['card_bg'])
        self.emotion_label.pack(side='right')
        
        # Emotion confidence
        emotion_conf_frame = tk.Frame(emotion_content, bg=self.colors['card_bg'])
        emotion_conf_frame.pack(fill='x', pady=(10, 0))
        
        tk.Label(emotion_conf_frame, text="Confidence:", 
                font=('Segoe UI', 11), fg=self.colors['text_dark'], 
                bg=self.colors['card_bg']).pack(side='left')
        
        self.emotion_conf_var = tk.StringVar(value="-")
        self.emotion_conf_label = tk.Label(emotion_conf_frame, textvariable=self.emotion_conf_var,
                                          font=('Segoe UI', 12, 'bold'), 
                                          fg=self.colors['success'], 
                                          bg=self.colors['card_bg'])
        self.emotion_conf_label.pack(side='right')
        
        # Emotion source
        emotion_source_frame = tk.Frame(emotion_content, bg=self.colors['card_bg'])
        emotion_source_frame.pack(fill='x', pady=(10, 0))
        
        tk.Label(emotion_source_frame, text="Analysis Source:", 
                font=('Segoe UI', 11), fg=self.colors['text_dark'], 
                bg=self.colors['card_bg']).pack(side='left')
        
        self.emotion_source_var = tk.StringVar(value="-")
        self.emotion_source_label = tk.Label(emotion_source_frame, textvariable=self.emotion_source_var,
                                            font=('Segoe UI', 10), 
                                            fg=self.colors['text_light'], 
                                            bg=self.colors['card_bg'])
        self.emotion_source_label.pack(side='right')
        
    def create_summary_card(self, parent):
        """Create summary results card"""
        summary_card = tk.Frame(parent, bg=self.colors['card_bg'], 
                               relief='raised', bd=2)
        summary_card.pack(fill='x')
        
        # Summary header
        summary_header = tk.Frame(summary_card, bg=self.colors['secondary'], height=50)
        summary_header.pack(fill='x')
        summary_header.pack_propagate(False)
        
        summary_title = tk.Label(summary_header, text="🎯 Analysis Summary", 
                                font=('Segoe UI', 14, 'bold'), 
                                fg='white', bg=self.colors['secondary'])
        summary_title.pack(pady=15)
        
        # Summary content
        summary_content = tk.Frame(summary_card, bg=self.colors['card_bg'])
        summary_content.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.summary_var = tk.StringVar(value="🚀 Ready to analyze your pet's mood!\nUpload an image or video to get started.")
        self.summary_label = tk.Label(summary_content, textvariable=self.summary_var,
                                     font=('Segoe UI', 12), 
                                     fg=self.colors['text_dark'], 
                                     bg=self.colors['card_bg'],
                                     wraplength=300, justify='center')
        self.summary_label.pack(pady=10)
        
    def create_status_bar(self):
        """Create modern status bar"""
        status_frame = tk.Frame(self.root, bg=self.colors['text_dark'], height=35)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="🟢 Ready - Upload your pet's photo or video to begin analysis")
        status_label = tk.Label(status_frame, textvariable=self.status_var,
                               font=('Segoe UI', 10), 
                               fg='white', bg=self.colors['text_dark'])
        status_label.pack(pady=8)
        
    def get_emotion_emoji(self, emotion):
        """Get emoji for emotion"""
        emotion_emojis = {
            'happy': '😊', 'sad': '😢', 'angry': '😠', 'surprised': '😲',
            'fear': '😨', 'disgust': '🤢', 'neutral': '😐', 'joy': '😄',
            'calm': '😌', 'excited': '🤩'
        }
        return emotion_emojis.get(emotion.lower(), '😊')
        
    def get_animal_emoji(self, animal):
        """Get emoji for animal"""
        animal_emojis = {
            'cat': '🐱', 'dog': '🐶', 'bird': '🐦', 'rabbit': '🐰',
            'hamster': '🐹', 'fish': '🐠'
        }
        return animal_emojis.get(animal.lower(), '🐾')
        
    def display_preview(self, file_path):
        """Display preview with modern styling"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            
            if file_ext in ['.jpg', '.jpeg', '.png']:
                # Display image preview
                image = Image.open(file_path)
                orig_width, orig_height = image.size
                
                # Resize for preview
                max_size = 350
                if orig_width > max_size or orig_height > max_size:
                    ratio = min(max_size / orig_width, max_size / orig_height)
                    new_width = int(orig_width * ratio)
                    new_height = int(orig_height * ratio)
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Add rounded corners effect
                photo = ImageTk.PhotoImage(image)
                self.preview_label.configure(image=photo, text="", bg='white')
                self.preview_label.image = photo
                
                self.file_info_var.set(f"📸 Image: {orig_width}×{orig_height}px • {file_size_mb:.1f}MB")
                
            elif file_ext in ['.mp4', '.avi', '.mov']:
                # Extract first frame for preview
                cap = cv2.VideoCapture(file_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                success, frame = cap.read()
                cap.release()
                
                if success:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    
                    # Resize for preview
                    max_size = 350
                    if width > max_size or height > max_size:
                        ratio = min(max_size / width, max_size / height)
                        new_width = int(width * ratio)
                        new_height = int(height * ratio)
                        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    photo = ImageTk.PhotoImage(image)
                    self.preview_label.configure(image=photo, text="", bg='white')
                    self.preview_label.image = photo
                    
                    self.file_info_var.set(f"🎥 Video: {width}×{height}px • {duration:.1f}s • {file_size_mb:.1f}MB")
                else:
                    self.preview_label.configure(image="", text="❌ Could not load video preview", 
                                               bg='#ffebee', fg='#c62828')
                    self.file_info_var.set(f"🎥 Video file: {file_size_mb:.1f}MB")
            
        except Exception as e:
            self.preview_label.configure(image="", text="⚠️ Preview not available", 
                                       bg='#fff3e0', fg='#ef6c00')
            self.file_info_var.set(f"Error: {str(e)}")
            
    def reset_display(self):
        """Reset display to initial state"""
        self.preview_label.configure(image="", text="🎯 Upload an image or video\nto see preview here", 
                                   fg=self.colors['text_light'])
        self.preview_label.image = None
        self.file_info_var.set("")
        self.animal_var.set("Waiting for upload...")
        self.animal_conf_var.set("-")
        self.emotion_var.set("Analyzing...")
        self.emotion_conf_var.set("-")
        self.emotion_source_var.set("-")
        self.summary_var.set("🚀 Ready to analyze your pet's mood!\nUpload an image or video to get started.")
        
    def update_confidence_color(self, confidence, label_widget):
        """Update confidence label color based on value"""
        if confidence >= 0.8:
            label_widget.configure(fg=self.colors['success'])  # Green for high confidence
        elif confidence >= 0.6:
            label_widget.configure(fg=self.colors['warning'])  # Orange for medium confidence
        else:
            label_widget.configure(fg=self.colors['accent'])   # Red for low confidence
    
    def save_results_to_firebase(self, result, file_path):
        """Save detection results to Firebase"""
        try:
            # Get dummy values for loss and accuracy (you can replace these with actual values if available)
            loss = 0.0  # Placeholder - replace with actual loss if available
            accuracy = max(result['animal_confidence'], result['emotion_confidence'])  # Use best confidence as accuracy
            
            # Determine audio path
            audio_path = None
            if result['source'] == 'video':
                # For videos, we would have extracted audio temporarily
                # Since we cleaned it up, we'll pass the original video path
                audio_path = file_path if file_path.endswith(('.mp4', '.avi', '.mov')) else None
            
            # Save to Firebase
            save_to_firebase(
                
                animal=result['animal'],
                emotion=result['emotion'],
                confidence_animal=result['animal_confidence'],
                confidence_emotion=result['emotion_confidence'],
                loss=loss,
                accuracy=accuracy,
                video_path=file_path,
                audio_path=audio_path
            )
            
            print("✅ Results saved to Firebase successfully")
            
        except Exception as e:
            print(f"❌ Failed to save to Firebase: {e}")
            messagebox.showwarning("Firebase Error", 
                                 f"Results processed successfully but failed to save to Firebase:\n{str(e)}")
            
    def upload_file(self):
        """Handle file upload with modern UI feedback"""
        file_path = filedialog.askopenfilename(
            title="🐾 Select Your Pet's Photo or Video",
            filetypes=[
                ("All Supported", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov"),
                ("Images", "*.jpg *.jpeg *.png"),
                ("Videos", "*.mp4 *.avi *.mov"),
                ("All Files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        # Reset display
        self.reset_display()
        
        # Show loading state
        self.status_var.set("📁 Loading file...")
        self.upload_btn.configure(text="⏳ Loading...", state='disabled')
        self.root.update()
        
        # Display preview
        self.display_preview(file_path)
        
        try:
            self.status_var.set("🔍 Analyzing your pet... This may take a moment!")
            self.animal_var.set("🔍 Detecting animal...")
            self.emotion_var.set("🧠 Analyzing emotions...")
            self.root.update()
            
            # Determine file type and predict
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in ['.jpg', '.jpeg', '.png']:
                result = self.detector.predict_from_image(file_path)
            elif file_ext in ['.mp4', '.avi', '.mov']:
                result = self.detector.predict_from_video(file_path)
            else:
                messagebox.showerror("❌ Unsupported Format", 
                                   "Please select a supported file format:\n• Images: JPG, PNG\n• Videos: MP4, AVI, MOV")
                self.status_var.set("❌ Unsupported file format")
                return
            
            # Display results with emojis and colors
            animal_emoji = self.get_animal_emoji(result['animal'])
            emotion_emoji = self.get_emotion_emoji(result['emotion'])
            
            self.animal_var.set(f"{animal_emoji} {result['animal'].title()}")
            self.animal_conf_var.set(f"{result['animal_confidence']:.1%}")
            self.update_confidence_color(result['animal_confidence'], self.animal_conf_label)
            
            self.emotion_var.set(f"{emotion_emoji} {result['emotion'].title()}")
            self.emotion_conf_var.set(f"{result['emotion_confidence']:.1%}")
            self.update_confidence_color(result['emotion_confidence'], self.emotion_conf_label)
            
            if result['source'] == 'video':
                source_icons = {'audio': '🔊', 'visual': '👁️', 'none': '❓'}
                source_icon = source_icons.get(result['emotion_source'], '📊')
                self.emotion_source_var.set(f"{source_icon} {result['emotion_source'].title()}")
            else:
                self.emotion_source_var.set("👁️ Visual Analysis")
            
            # Create engaging summary
            confidence_level = "High" if (result['animal_confidence'] > 0.8 and result['emotion_confidence'] > 0.6) else \
                             "Good" if (result['animal_confidence'] > 0.6 and result['emotion_confidence'] > 0.4) else "Fair"
            
            summary_text = f"🎉 Analysis Complete!\n\n"
            summary_text += f"Your {result['animal']} appears to be feeling {result['emotion']}!\n\n"
            summary_text += f"📊 Confidence Level: {confidence_level}\n"
            
            if result['source'] == 'video':
                summary_text += f"🔍 Used {result['emotion_source']} analysis for emotion detection"
            
            self.summary_var.set(summary_text)
            self.status_var.set(f"✅ Analysis complete! Found a {result['emotion']} {result['animal']}")
            
            # Save results to Firebase
            self.save_results_to_firebase(result, file_path)
            
            # Show detailed popup for videos
            if result['source'] == 'video' and 'audio_emotion' in result:
                self.show_detailed_video_results(result)
                
        except Exception as e:
            error_msg = f"❌ Analysis Failed\n\nError: {str(e)}\n\nPlease try with a different file or check that all model files are present."
            messagebox.showerror("Analysis Error", error_msg)
            self.status_var.set("❌ Analysis failed - please try again")
            print(f"Error details: {e}")
            
        finally:
            # Re-enable upload button
            self.upload_btn.configure(text="🔍 Select Image or Video", state='normal')
            
    def show_detailed_video_results(self, result):
        """Show detailed results popup for video analysis"""
        # Create custom dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("🎬 Detailed Video Analysis")
        dialog.geometry("500x400")
        dialog.configure(bg=self.colors['background'])
        dialog.resizable(False, False)
        
        # Center the dialog
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Header
        header_frame = tk.Frame(dialog, bg=self.colors['primary'], height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(header_frame, text="🎬 Video Analysis Results", 
                               font=('Segoe UI', 18, 'bold'), 
                               fg='white', bg=self.colors['primary'])
        header_label.pack(pady=25)
        
        # Content
        content_frame = tk.Frame(dialog, bg=self.colors['background'])
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Main results
        main_results = tk.Frame(content_frame, bg=self.colors['card_bg'], relief='raised', bd=2)
        main_results.pack(fill='x', pady=(0, 15))
        
        main_title = tk.Label(main_results, text="🎯 Final Results", 
                             font=('Segoe UI', 14, 'bold'), 
                             fg=self.colors['text_dark'], bg=self.colors['card_bg'])
        main_title.pack(pady=(15, 10))
        
        animal_emoji = self.get_animal_emoji(result['animal'])
        emotion_emoji = self.get_emotion_emoji(result['emotion'])
        
        main_text = f"{animal_emoji} Animal: {result['animal'].title()} ({result['animal_confidence']:.1%})\n"
        main_text += f"{emotion_emoji} Emotion: {result['emotion'].title()} ({result['emotion_confidence']:.1%})\n"
        main_text += f"📊 Primary Source: {result['emotion_source'].title()}"
        
        main_label = tk.Label(main_results, text=main_text, 
                             font=('Segoe UI', 12), 
                             fg=self.colors['text_dark'], bg=self.colors['card_bg'])
        main_label.pack(pady=(0, 15))
        
        # Detailed breakdown
        details_frame = tk.Frame(content_frame, bg=self.colors['card_bg'], relief='raised', bd=2)
        details_frame.pack(fill='x', pady=(0, 15))
        
        details_title = tk.Label(details_frame, text="📋 Analysis Breakdown", 
                                font=('Segoe UI', 14, 'bold'), 
                                fg=self.colors['text_dark'], bg=self.colors['card_bg'])
        details_title.pack(pady=(15, 10))
        
        audio_info = result['audio_emotion']
        visual_info = result['visual_emotion']
        
        details_text = f"🔊 Audio Analysis: {audio_info['label'].title() if audio_info['label'] else 'Not Available'} "
        details_text += f"({audio_info['confidence']:.1%})\n\n"
        details_text += f"👁️ Visual Analysis: {visual_info['label'].title() if visual_info['label'] else 'Not Available'} "
        details_text += f"({visual_info['confidence']:.1%})\n\n"
        details_text += "ℹ️ The system automatically selects the analysis method with higher confidence."
        
        details_label = tk.Label(details_frame, text=details_text, 
                                font=('Segoe UI', 11), 
                                fg=self.colors['text_dark'], bg=self.colors['card_bg'],
                                justify='left')
        details_label.pack(pady=(0, 15), padx=15)
        
        # Close button
        close_btn = tk.Button(content_frame, text="✅ Got it!", 
                             command=dialog.destroy,
                             font=('Segoe UI', 12, 'bold'),
                             bg=self.colors['success'], fg='white',
                             relief='flat', padx=30, pady=10,
                             cursor='hand2')
        close_btn.pack(pady=10)
        
    def run(self):
        """Start the modern GUI"""
        # Center window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
        
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = ModernAnimalEmotionGUI()
        app.run()
    except Exception as e:
        error_window = tk.Tk()
        error_window.title("🚨 Startup Error")
        error_window.geometry("600x300")
        error_window.configure(bg='#ffebee')
        
        error_label = tk.Label(error_window, 
                              text=f"❌ Failed to start PetMood Analyzer\n\n"
                                   f"Error: {str(e)}\n\n"
                                   f"Please ensure all required files are present:\n"
                                   f"📁 backend/model_animal_best.pth\n"
                                   f"📁 backend/model_emotion.pth\n" 
                                   f"📁 backend/audio_emotion.pth\n"
                                   f"📁 backend/emotion_classes.txt\n"
                                   f"📁 backend/emotion_classes_audio.txt\n"
                                   f"📁 firebase_utils.py\n"
                                   f"📁 firebase_config.json",
                              font=('Segoe UI', 11),
                              fg='#c62828', bg='#ffebee',
                              justify='center')
        error_label.pack(expand=True)
        
        error_window.mainloop()