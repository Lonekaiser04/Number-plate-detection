# License Plate Recognition System

An intelligent license plate detection and recognition system with blacklist verification using YOLOv8, Tesseract OCR, and EasyOCR.

## Features

- Vehicle and license plate detection using YOLOv8
- Multi-engine OCR (Tesseract + EasyOCR) for accurate text extraction
- Advanced image preprocessing (contrast enhancement, noise reduction, adaptive thresholding)
- Blacklist checking with exact and fuzzy matching
- GUI application with image preview and debug view
- Real-time security alerts for blacklisted vehicles

## Installation

### Prerequisites

- Python 3.8+
- Tesseract OCR installed on your system

### Install Tesseract OCR

**Windows:** Download from https://github.com/UB-Mannheim/tesseract/wiki
**Linux:** `sudo apt-get install tesseract-ocr`
**macOS:** `brew install tesseract`

### Install Python Dependencies

```bash
pip install opencv-python pytesseract pandas ultralytics pillow numpy easyocr

license-plate-system/
├── Main.py              # Main application
├── blacklist.csv        # Blacklist database
├── Yolov8.pt             # Model
└── README.md           # Documentation

python Main.py
