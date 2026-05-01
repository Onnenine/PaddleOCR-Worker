import os
import sys
import argparse
from paddleocr import PaddleOCR

# Khởi tạo PaddleOCR (lang='ch' cho tiếng Trung)
# use_angle_cls=True để tự động xoay chữ nếu ảnh bị nghiêng
ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False, show_log=False)

def process_ocr(image_path):
    print(f"--- Đang xử lý: {image_path} ---")
    result = ocr.ocr(image_path, cls=True)
    
    if not result or result[0] is None:
        print("Không tìm thấy chữ nào.")
        return ""

    texts = []
    for line in result[0]:
        text = line[1][0]
        confidence = line[1][1]
        if confidence > 0.5: # Chỉ lấy chữ có độ tin cậy trên 50%
            texts.append(text)
    
    full_text = " ".join(texts)
    print(f"Kết quả: {full_text}")
    return full_text

if __name__ == "__main__":
    # Ví dụ: chạy python run_ocr.py test.jpg
    if len(sys.argv) > 1:
        process_ocr(sys.argv[1])
    else:
        print("Vui lòng cung cấp đường dẫn ảnh!")
