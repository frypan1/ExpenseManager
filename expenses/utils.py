# expenses/utils.py
import pytesseract
import numpy as np
import cv2

def perform_ocr(image_path):
    try:
            # Load the image
        image = cv2.imread(image_path)

        # Step 1: Resize
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Step 2: Convert to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 3: Binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 4: Noise Removal
        denoised = cv2.medianBlur(binary, 3)

        # Step 5: Deskewing
        coords = np.column_stack(np.where(denoised > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = denoised.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        denoised = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Step 6: OCR
        text = pytesseract.image_to_string(denoised)
        return text
    except Exception as e:
        return str(e)