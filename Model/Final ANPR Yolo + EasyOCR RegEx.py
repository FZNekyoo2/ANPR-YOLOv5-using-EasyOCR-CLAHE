import os
import cv2
import torch
import easyocr
import re

# ======== Konfigurasi ========
input_folder = 'D:/Final Projek PCD/Gambar Projek PCD Data'
output_folder = 'D:/Final Projek PCD/Final Regex Format'
yolo_model_path = 'D:/Final Projek PCD/finalmodel.pt'
os.makedirs(output_folder, exist_ok=True)

# ======== Load YOLOv5 dan EasyOCR ========
reader = easyocr.Reader(['id'], gpu=True)
yolo_model = torch.hub.load('D:/Libraries Coding/yolov5', 'custom', path=yolo_model_path, source='local')
yolo_model.conf = 0.1

# ======== Proses Setiap Gambar ========
hasil_txt = os.path.join(output_folder, 'hasil_ocr.txt')
with open(hasil_txt, 'w') as file_out:
    for img_name in os.listdir(input_folder):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)
        h, w = image.shape[:2]

        # Deteksi plat dengan YOLO
        results = yolo_model(image)
        detections = results.xyxy[0]

        for i, (*box, conf, cls) in enumerate(detections):
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1 - 10)
            y1 = max(0, y1 - 10)
            x2 = min(w, x2 + 10)
            y2 = min(h, y2 + 10)

            crop = image[y1:y2, x1:x2]

            # ======== Preprocessing ========
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # ======== Simpan Gambar Hasil Preprocessing ========
            pre_output_path = os.path.join(output_folder, f"pre_{i}_{img_name}")
            cv2.imwrite(pre_output_path, blurred)

            # ======== OCR dengan EasyOCR ========
            result = reader.readtext(blurred)

            if result:
                # Urutkan berdasarkan posisi X (kiri ke kanan)
                result_sorted = sorted(result, key=lambda x: x[0][0][0])
                texts = [r[1] for r in result_sorted]
                confs = [r[2] for r in result_sorted]

                raw_text = " ".join(texts)
                cleaned_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())  # Hilangkan selain huruf & angka

                # Ekstrak bagian huruf dan angka
                letters = re.findall(r'[A-Z]+', cleaned_text)
                digits = re.findall(r'\d+', cleaned_text)

                if letters and digits:
                    ll = letters[0][:2]
                    nnnn = digits[0][:4]
                    lll = ''.join(letters[1:])[:3] if len(letters) > 1 else ''
                    plate_text = f"{ll} {nnnn} {lll}"
                    conf_score = min(confs)

                    # Tampilkan dan simpan hasil
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
                    file_out.write(f"{img_name} -> {plate_text} (conf={conf_score:.2f})\n")
                else:
                    print(f"[!] Gagal ekstrak dari OCR: {cleaned_text}")

        # Simpan gambar hasil anotasi
        output_img = os.path.join(output_folder, f"result_{img_name}")
        cv2.imwrite(output_img, image)
        print(f"[✔] Disimpan: {output_img}")

print(f"[✔] Semua hasil disimpan di {hasil_txt}")
