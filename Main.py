import cv2
import pytesseract
import pandas as pd
from ultralytics import YOLO
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import numpy as np
import re
import time
import easyocr
from difflib import SequenceMatcher

class LicensePlateSystem:
    def __init__(self):
        try:
            # Use a specialized license plate detection model if available
            self.model = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
        
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'])
        
        self.blacklist_file = "blacklist.csv"
        self.ensure_blacklist_exists()
        self.blacklist = self.load_blacklist()
        
        # Common OCR misreadings mapping
        self.ocr_corrections = {
            'O': '0', 'I': '1', 'Z': '2', 'S': '5', 
            'D': '0', 'G': '6', 'T': '7', ' ': '', '.': '', '-': ''
        }

    def ensure_blacklist_exists(self):
        if not os.path.exists(self.blacklist_file):
            sample_data = {
                "PlateNumber": ["JK08D4356", "JK01AB1234", "JK14SU3550", "JK08XP1434"],
                "Reason": ["Traffic Violation", "Stolen Vehicle", "Duplicate Number Plate", "Unregistered Vehicle"]
            }
            pd.DataFrame(sample_data).to_csv(self.blacklist_file, index=False)

    def load_blacklist(self):
        try:
            df = pd.read_csv(self.blacklist_file)
            df["PlateNumber"] = df["PlateNumber"].str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
            return df
        except Exception as e:
            print(f"Error loading blacklist: {e}")
            return pd.DataFrame(columns=["PlateNumber", "Reason"])

    def detect_vehicles(self, image_path):
        if self.model is None:
            return []
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            results = self.model(img, classes=[2, 3, 5, 7], conf=0.3)
            
            vehicle_regions = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        padding = 20
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(img.shape[1], x2 + padding)
                        y2 = min(img.shape[0], y2 + padding)
                        
                        vehicle_region = img[y1:y2, x1:x2]
                        if vehicle_region.size > 0:
                            vehicle_regions.append(vehicle_region)
            
            return vehicle_regions
        except Exception as e:
            print(f"Error in vehicle detection: {e}")
            return []

    def find_plate_regions(self, vehicle_img):
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise while keeping edges sharp
            bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            
            plate_candidates = []
            
            for contour in contours:
                # Get rectangle bounding contour
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                area = w * h
                
                # Check if region could be a license plate
                if (2.0 <= aspect_ratio <= 5.0 and area > 1000 and w > 60 and h > 20):
                    plate_region = vehicle_img[y:y+h, x:x+w]
                    
                    # Additional check for text-like regions
                    plate_gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
                    sobelx = cv2.Sobel(plate_gray, cv2.CV_8U, 1, 0, ksize=3)
                    sobel_density = np.sum(sobelx > 0) / (w * h)
                    
                    if sobel_density > 0.1:
                        plate_candidates.append(plate_region)
            
            return plate_candidates
        except Exception as e:
            print(f"Error finding plate regions: {e}")
            return []

    def detect_plates(self, image_path):
        plates = []
        
        vehicle_regions = self.detect_vehicles(image_path)
        
        for vehicle_region in vehicle_regions:
            plate_regions = self.find_plate_regions(vehicle_region)
            plates.extend(plate_regions)
        
        if not plates:
            img = cv2.imread(image_path)
            if img is not None:
                plate_regions = self.find_plate_regions(img)
                plates.extend(plate_regions)
        
        return plates

    def advanced_preprocess(self, plate_img):
        """Advanced preprocessing for better OCR accuracy"""
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Resize if too small
        height, width = gray.shape
        if width < 100 or height < 30:
            scale_x = 300 / width
            scale_y = 100 / height
            scale = min(scale_x, scale_y)
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(morph)
        
        # Apply sharpening filter
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        return sharpened

    def correct_common_ocr_errors(self, text):
        """Correct common OCR misreadings"""
        corrected = text.upper()
        for wrong, right in self.ocr_corrections.items():
            corrected = corrected.replace(wrong, right)
        return corrected

    def validate_plate_format(self, text):
        """Validate if text looks like a license plate"""
        # Remove all non-alphanumeric characters
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Check if it has a reasonable length for a license plate
        if len(cleaned) < 6 or len(cleaned) > 12:
            return False
            
        # Check if it has at least some digits and some letters
        has_digits = any(char.isdigit() for char in cleaned)
        has_letters = any(char.isalpha() for char in cleaned)
        
        return has_digits and has_letters

    def extract_plate_text(self, plate_img):
        """Advanced OCR approach with multiple engines and strategies"""
        try:
            # Get enhanced preprocessed image
            processed_img = self.advanced_preprocess(plate_img)
            
            # Try multiple OCR approaches
            results = []
            
            # Approach 1: Tesseract with multiple configurations
            tesseract_configs = [
                '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ]
            
            for config in tesseract_configs:
                try:
                    # Get OCR data with confidence
                    ocr_data = pytesseract.image_to_data(processed_img, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Extract text with high confidence
                    text_parts = []
                    conf_parts = []
                    
                    for i in range(len(ocr_data['text'])):
                        if int(ocr_data['conf'][i]) > 60:  # Only consider high confidence results
                            text = ocr_data['text'][i].strip()
                            if text and len(text) > 1:
                                text_parts.append(text)
                                conf_parts.append(int(ocr_data['conf'][i]))
                    
                    if text_parts:
                        candidate_text = ' '.join(text_parts)
                        candidate_confidence = sum(conf_parts) / len(conf_parts) if conf_parts else 0
                        
                        # Clean the text
                        cleaned = re.sub(r'[^A-Z0-9]', '', candidate_text.upper())
                        
                        # Correct common OCR errors
                        corrected = self.correct_common_ocr_errors(cleaned)
                        
                        if self.validate_plate_format(corrected):
                            results.append((corrected, candidate_confidence, "Tesseract"))
                            
                except Exception as e:
                    print(f"Tesseract config error: {e}")
                    continue
            
            # Approach 2: EasyOCR
            try:
                easy_results = self.reader.readtext(processed_img, detail=1)
                for (bbox, text, confidence) in easy_results:
                    if confidence > 0.6:  # Confidence threshold
                        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                        corrected = self.correct_common_ocr_errors(cleaned)
                        if self.validate_plate_format(corrected):
                            results.append((corrected, confidence * 100, "EasyOCR"))
            except Exception as e:
                print(f"EasyOCR error: {e}")
            
            # Approach 3: Simple thresholding + Tesseract
            try:
                # Convert to grayscale
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                
                # Apply simple threshold
                _, simple_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Resize
                simple_resized = cv2.resize(simple_thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                
                # Try OCR
                text = pytesseract.image_to_string(simple_resized, config='--oem 3 --psm 8')
                cleaned = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
                corrected = self.correct_common_ocr_errors(cleaned)
                
                if self.validate_plate_format(corrected):
                    # Estimate confidence based on length and character diversity
                    confidence = min(80 + len(corrected) * 2, 95)
                    results.append((corrected, confidence, "Simple Threshold"))
            except Exception as e:
                print(f"Simple threshold OCR error: {e}")
            
            # If we have results, select the best one
            if results:
                # Sort by confidence
                results.sort(key=lambda x: x[1], reverse=True)
                best_text, best_confidence, method = results[0]
                
                print(f"Selected text: {best_text} with confidence {best_confidence} from {method}")
                return best_text
            
            return "UNREADABLE"
            
        except Exception as e:
            print(f"Error in OCR: {e}")
            return "ERROR"

    def check_blacklist(self, plate_text):
        if not plate_text or plate_text in ["UNREADABLE", "ERROR"]:
            return False, ""
        
        try:
            plate_text_clean = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
            
            # First try exact match
            matches = self.blacklist[self.blacklist["PlateNumber"] == plate_text_clean]
            if not matches.empty:
                return True, matches.iloc[0]["Reason"]
            
            # Then try fuzzy matching for common OCR errors
            for _, row in self.blacklist.iterrows():
                blacklisted_plate = row["PlateNumber"]
                similarity = SequenceMatcher(None, plate_text_clean, blacklisted_plate).ratio()
                if similarity > 0.8:  # 80% similarity threshold
                    return True, row["Reason"]
            
            return False, ""
            
        except Exception as e:
            print(f"Error checking blacklist: {e}")
            return False, ""

# Configure Tesseract path
try:
    if os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    elif os.system('which tesseract') == 0:
        pass
    else:
        print("Warning: Tesseract not found. Please install Tesseract OCR.")
except:
    pass

class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Recognition System")
        self.root.geometry("1200x800")
        
        self.lp_system = LicensePlateSystem()
        self.setup_ui()
        
        self.image_path = ""
        self.plate_images = []
        self.detection_results = []

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(main_frame, text="License Plate Recognition System", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_open = ttk.Button(control_frame, text="Open Image", command=self.open_image)
        self.btn_open.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_detect = ttk.Button(control_frame, text="Detect Plates", command=self.detect_plates, state=tk.DISABLED)
        self.btn_detect.pack(side=tk.LEFT, padx=5)
        
        self.btn_debug = ttk.Button(control_frame, text="Debug View", command=self.show_debug, state=tk.DISABLED)
        self.btn_debug.pack(side=tk.LEFT, padx=5)
        
        # Image display
        img_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="5")
        img_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.canvas = tk.Canvas(img_frame, bg="white")
        v_scrollbar = ttk.Scrollbar(img_frame, orient="vertical", command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(img_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Results
        result_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="5")
        result_frame.pack(fill=tk.X, pady=(0, 10))
        
        columns = ("Index", "Plate", "Status", "Reason")
        self.tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=6)
        
        self.tree.heading("Index", text="#")
        self.tree.heading("Plate", text="License Plate")
        self.tree.heading("Status", text="Status")
        self.tree.heading("Reason", text="Reason")
        
        self.tree.column("Index", width=50)
        self.tree.column("Plate", width=150)
        self.tree.column("Status", width=100)
        self.tree.column("Reason", width=300)
        
        tree_scroll = ttk.Scrollbar(result_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_image()
            self.btn_detect.config(state=tk.NORMAL)
            self.btn_debug.config(state=tk.DISABLED)
            self.clear_results()
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")

    def display_image(self):
        try:
            pil_image = Image.open(self.image_path)
            
            # Resize for display
            display_width, display_height = 800, 600
            img_width, img_height = pil_image.size
            scale = min(display_width/img_width, display_height/img_height, 1.0)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            display_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(display_image)
            
            self.canvas.config(scrollregion=(0, 0, new_width, new_height))
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")

    def clear_results(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.plate_images.clear()
        self.detection_results.clear()

    def detect_plates(self):
        if not self.image_path:
            return
        
        try:
            self.btn_detect.config(state=tk.DISABLED)
            self.status_var.set("Processing image...")
            self.root.update()
            
            self.clear_results()
            
            # Detect plates
            self.plate_images = self.lp_system.detect_plates(self.image_path)
            
            if not self.plate_images:
                messagebox.showinfo("No Plates Found", "No license plates detected in the image.")
                self.status_var.set("Ready - No plates detected")
                return
            
            # Process each plate
            results = []
            for i, plate_img in enumerate(self.plate_images):
                try:
                    plate_text = self.lp_system.extract_plate_text(plate_img)
                    is_blacklisted, reason = self.lp_system.check_blacklist(plate_text)
                    
                    if plate_text in ["UNREADABLE", "ERROR"]:
                        status = "UNREADABLE"
                        reason = "Could not extract text clearly"
                    else:
                        status = "BLACKLISTED" if is_blacklisted else "CLEAN"
                    
                    results.append({
                        'index': i + 1,
                        'plate_text': plate_text,
                        'status': status,
                        'reason': reason,
                        'is_blacklisted': is_blacklisted,
                        'plate_img': plate_img
                    })
                    
                except Exception as e:
                    results.append({
                        'index': i + 1,
                        'plate_text': "ERROR",
                        'status': "ERROR",
                        'reason': f"Processing error: {str(e)}",
                        'is_blacklisted': False,
                        'plate_img': None
                    })
            
            # Store results for debug view
            self.detection_results = results
            
            # Remove duplicates and keep the best result for each plate
            unique_results = self.remove_duplicate_results(results)
            
            # Update results table
            for result in unique_results:
                if result['is_blacklisted']:
                    tags = ('blacklisted',)
                elif result['status'] == "CLEAN":
                    tags = ('clean',)
                else:
                    tags = ('unreadable',)
                
                self.tree.insert("", "end", values=(
                    result['index'],
                    result['plate_text'],
                    result['status'],
                    result['reason']
                ), tags=tags)
            
            # Configure tag colors
            self.tree.tag_configure('blacklisted', background='#ffebee', foreground='#c62828')
            self.tree.tag_configure('clean', background='#e8f5e8', foreground='#2e7d32')
            self.tree.tag_configure('unreadable', background='#fff3e0', foreground='#f57c00')
            
            # Show alerts
            blacklisted_plates = [r for r in unique_results if r['is_blacklisted']]
            if blacklisted_plates:
                alert_msg = "SECURITY ALERT\n\nBlacklisted plates detected:\n\n"
                for plate in blacklisted_plates:
                    alert_msg += f"• {plate['plate_text']}: {plate['reason']}\n"
                messagebox.showwarning("Security Alert", alert_msg)
            
            self.status_var.set(f"Detection complete - {len(unique_results)} plates processed")
            self.btn_debug.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Detection Error", f"Error: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
        
        finally:
            self.btn_detect.config(state=tk.NORMAL)

    def remove_duplicate_results(self, results):
        """Remove duplicate plate detections and keep the best result"""
        unique_plates = {}
        
        for result in results:
            plate_text = result['plate_text']
            
            # Skip UNREADABLE and ERROR results if we have better ones
            if plate_text in ["UNREADABLE", "ERROR"]:
                if plate_text not in unique_plates:
                    unique_plates[plate_text] = result
                continue
            
            # For valid plates, keep the one with highest confidence (assumed based on status)
            if plate_text not in unique_plates:
                unique_plates[plate_text] = result
            else:
                # Prefer BLACKLISTED over CLEAN (more important to alert)
                if result['is_blacklisted'] and not unique_plates[plate_text]['is_blacklisted']:
                    unique_plates[plate_text] = result
        
        return list(unique_plates.values())

    def show_debug(self):
        if not self.detection_results:
            messagebox.showinfo("No Debug Data", "No plates detected to debug.")
            return
        
        debug_window = tk.Toplevel(self.root)
        debug_window.title("Plate Debug Information")
        debug_window.geometry("1000x700")
        
        notebook = ttk.Notebook(debug_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for i, result in enumerate(self.detection_results):
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=f"Plate {i+1}")
            
            # Original plate image
            try:
                if result['plate_img'] is not None:
                    plate_rgb = cv2.cvtColor(result['plate_img'], cv2.COLOR_BGR2RGB)
                    plate_pil = Image.fromarray(plate_rgb)
                    
                    display_size = (400, 200)
                    plate_pil.thumbnail(display_size, Image.Resampling.LANCZOS)
                    plate_photo = ImageTk.PhotoImage(plate_pil)
                    
                    ttk.Label(frame, text="Original Detected Region:", font=("Arial", 12, "bold")).pack(pady=10)
                    
                    orig_img_label = tk.Label(frame, image=plate_photo)
                    orig_img_label.image = plate_photo
                    orig_img_label.pack(pady=5)
                    
                    # Processed image
                    processed_img = self.lp_system.advanced_preprocess(result['plate_img'])
                    if processed_img is not None:
                        proc_pil = Image.fromarray(processed_img)
                        proc_pil.thumbnail(display_size, Image.Resampling.LANCZOS)
                        proc_photo = ImageTk.PhotoImage(proc_pil)
                        
                        ttk.Label(frame, text="Processed for OCR:", font=("Arial", 12, "bold")).pack(pady=10)
                        
                        proc_img_label = tk.Label(frame, image=proc_photo)
                        proc_img_label.image = proc_photo
                        proc_img_label.pack(pady=5)
                
                # OCR Results
                plate_text = result['plate_text']
                is_blacklisted = result['is_blacklisted']
                reason = result['reason']
                
                results_frame = ttk.LabelFrame(frame, text="OCR Results", padding="10")
                results_frame.pack(fill=tk.X, padx=10, pady=20)
                
                ttk.Label(results_frame, text=f"Extracted Text: {plate_text}", 
                         font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=2)
                
                status_text = "BLACKLISTED" if is_blacklisted else "CLEAN"
                status_color = "red" if is_blacklisted else "green"
                
                status_label = tk.Label(results_frame, text=f"Status: {status_text}", 
                                      font=("Arial", 11, "bold"), fg=status_color)
                status_label.pack(anchor=tk.W, pady=2)
                
                if reason:
                    ttk.Label(results_frame, text=f"Reason: {reason}").pack(anchor=tk.W, pady=2)
                
                # Processing info
                info_frame = ttk.LabelFrame(frame, text="Processing Information", padding="10")
                info_frame.pack(fill=tk.X, padx=10, pady=10)
                
                if result['plate_img'] is not None:
                    ttk.Label(info_frame, text=f"Original Size: {result['plate_img'].shape[1]}x{result['plate_img'].shape[0]} pixels").pack(anchor=tk.W)
                    
                    # Image quality
                    gray_plate = cv2.cvtColor(result['plate_img'], cv2.COLOR_BGR2GRAY)
                    blur_score = cv2.Laplacian(gray_plate, cv2.CV_64F).var()
                    quality = "Good" if blur_score > 100 else "Fair" if blur_score > 50 else "Poor"
                    ttk.Label(info_frame, text=f"Image Quality: {quality} (Blur Score: {blur_score:.1f})").pack(anchor=tk.W)
                
            except Exception as e:
                ttk.Label(frame, text=f"Error displaying plate {i+1}: {str(e)}").pack(pady=20)

if __name__ == "__main__":
    missing_deps = []
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import pytesseract
    except ImportError:
        missing_deps.append("pytesseract")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        missing_deps.append("ultralytics")
    
    try:
        import easyocr
    except ImportError:
        missing_deps.append("easyocr")
    
    if missing_deps:
        print("Missing dependencies:")
        for dep in missing_deps:
            print(f"  pip install {dep}")
        exit(1)
    
    try:
        root = tk.Tk()
        app = LicensePlateApp(root)
        
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
        
        print("License Plate Recognition System Started")
        print("Ensure Tesseract OCR is installed and accessible")
        
        root.mainloop()
        
    except Exception as e:
        print(f"Failed to start application: {e}")
        input("Press Enter to exit...")