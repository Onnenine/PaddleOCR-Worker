from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Cho phép website của bạn gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Thay bằng domain website của bạn
    allow_methods=["*"],
    allow_headers=["*"],
)

# lang="vi" cho tiếng Việt, "en" cho tiếng Anh
ocr = PaddleOCR(use_angle_cls=True, lang="vi")

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image)
    
    result = ocr.ocr(img_array, cls=True)
    
    texts = []
    for line in result[0]:
        bbox, (text, confidence) = line
        texts.append({"text": text, "confidence": round(confidence, 4), "bbox": bbox})
    
    return {"results": texts, "full_text": " ".join([t["text"] for t in texts])}

@app.get("/health")
def health():
    return {"status": "ok"}
