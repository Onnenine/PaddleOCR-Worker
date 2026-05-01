import os
import sys
from paddleocr import PaddleOCR

# Khởi tạo
ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False, show_log=False)

def process_ocr(image_path):
    print(f"--- Processing: {image_path} ---")
    try:
        result = ocr.ocr(image_path, cls=True)
        if not result or result[0] is None:
            print("No text found.")
            return
        
        for line in result[0]:
            # line[1][0] là văn bản, line[1][1] là độ tự tin
            print(f"Detected: {line[1][0]} (Confidence: {line[1][1]})")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_ocr(sys.argv[1])
