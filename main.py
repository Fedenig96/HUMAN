import cv2
from ultralytics import YOLO
#https://www.swisstransfer.com/d/9b6e0f0c-6053-4f98-a724-41faa04b144c
# Carica il modello con segmentazione
model = YOLO("yolov8n.pt")

# Apri la webcam (0 = integrata, 1 = USB)
cap = cv2.VideoCapture(0)

# Crea una finestra a tutto schermo
cv2.namedWindow("YOLOv8 Person Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLOv8 Person Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for i in range(len(results.boxes)):
        box = results.boxes[i]
        cls_id = int(box.cls[0])

        if cls_id == 0:  # solo persone
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            # sagoma
            if results.masks is not None:
                mask = results.masks.data[i].cpu().numpy()
                mask = (mask * 255).astype('uint8')
                colored_mask = cv2.merge([mask, mask // 2, mask // 4])
                frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

    cv2.imshow("YOLOv8 Person Detection", frame)

    # Esci con ESC
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
