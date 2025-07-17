import cv2
import numpy as np
from ultralytics import YOLO
import joblib

# YOLO modelini yükle
yolo_model = YOLO("C:\\Users\\Lenovo\\Desktop\\best.pt")

# SVM modeli ve LabelEncoder'ı yükle
svm_model = joblib.load("svm_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Kamera başlat
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kamera açılamadı.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Görüntü alınamadı.")
        break

    # YOLO tahmini
    results = yolo_model.predict(source=frame, conf=0.7, verbose=False)

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        roi = frame[y1:y2, x1:x2]

        if roi.shape[0] == 0 or roi.shape[1] == 0:
            continue

        cv2.imshow("ROI Debug", roi)

        # ROI'deki ortalama renkleri hesapla
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        avg_r = np.mean(roi_rgb[:, :, 0])
        avg_g = np.mean(roi_rgb[:, :, 1])
        avg_b = np.mean(roi_rgb[:, :, 2])

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_h = np.mean(hsv[:, :, 0]) / 180.0  # H normalizasyonu
        avg_s = np.mean(hsv[:, :, 1]) / 255.0  # S normalizasyonu
        avg_v = np.mean(hsv[:, :, 2]) / 255.0  # V normalizasyonu

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        avg_l = np.mean(lab[:, :, 0])
        avg_a = np.mean(lab[:, :, 1]) - 128  # LAB offset düzeltmesi
        avg_b_lab = np.mean(lab[:, :, 2]) - 128  # LAB offset düzeltmesi

        # LOG: Terminale yazdır
        print("\n\n==============================", flush=True)
        print(f"📸 ROI Feature Vector:", flush=True)
        print(f"R={avg_r:.2f}, G={avg_g:.2f}, B={avg_b:.2f}", flush=True)
        print(f"H={avg_h:.2f}, S={avg_s:.2f}, V={avg_v:.2f}", flush=True)
        print(f"L={avg_l:.2f}, A={avg_a:.2f}, B_lab={avg_b_lab:.2f}", flush=True)
        print("==============================", flush=True)

        # Feature vector tam ve eğitimdekiyle aynı
        feature_vector = np.array([[avg_r, avg_g, avg_b, avg_h, avg_s, avg_v, avg_l, avg_a, avg_b_lab]])

        # SVM tahmini yap
        pred_class = svm_model.predict(feature_vector)
        pred_label = label_encoder.inverse_transform(pred_class)[0]

        print(f"✅ SVM sonucu: {pred_label}", flush=True)

        # Çizim ve etiket
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"Balon - {pred_label.strip()}"
        cv2.putText(frame, text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Balon ve Renk Sınıflandırması", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()