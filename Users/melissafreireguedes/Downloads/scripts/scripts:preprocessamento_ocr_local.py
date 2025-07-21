import pytesseract
import pdfplumber
from PIL import Image
import os
import cv2
import numpy as np
import glob

def extract_text_from_pdf(path):
    try:
        with pdfplumber.open(path) as pdf:
            texts = []
            for page in pdf.pages:
                text = page.extract_text()
                if not text or len(text.strip()) < 30:
                    print(f"[OCR] Página com conteúdo fraco, aplicando OCR em {path}")
                    image = page.to_image(resolution=300).original
                    ocr_text = pytesseract.image_to_string(image, lang='por')
                    texts.append(ocr_text)
                else:
                    texts.append(text)
            return "\n".join(texts)
    except Exception as e:
        print(f"[ERROR] Failed to process PDF: {path}", e)
        return ""

def extract_text_from_image(path):
    try:
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        scale_percent = 150
        width = int(thresh.shape[1] * scale_percent / 100)
        height = int(thresh.shape[0] * scale_percent / 100)
        resized = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_LINEAR)
        pil_img = Image.fromarray(resized)
        text = pytesseract.image_to_string(pil_img, lang="por")
        return text
    except Exception as e:
        print(f"[ERROR] Failed to process image: {path}", e)
        return ""

def extract_all_text(file_paths):
    full_text = ""
    for path in file_paths:
        name = os.path.basename(path)
        if name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(path)
        elif name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            text = extract_text_from_image(path)
        else:
            continue
        full_text += f"\n--- {name} ---\n{text}\n"
    return full_text

if __name__ == "__main__":
    file_paths = glob.glob("data/*")
    extracted_text = extract_all_text(file_paths)

    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print("Text extraction completed!")
