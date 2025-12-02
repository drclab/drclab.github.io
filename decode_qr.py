import cv2
import sys

image_path = "/home/cjduan/.gemini/antigravity/brain/79dd917d-e8e7-4be8-8663-d894e8f252c5/uploaded_image_1764702815157.png"
img = cv2.imread(image_path)

if img is None:
    print(f"Failed to load image from {image_path}")
    sys.exit(1)

detector = cv2.QRCodeDetector()
retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(img)

if retval:
    print("Decoded QR Codes:")
    for info in decoded_info:
        if info:
            print(f"- {info}")
else:
    print("No QR codes found.")
