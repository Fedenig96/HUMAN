import cv2
from ultralytics import YOLO

# Carica il modello con segmentazione
model = YOLO("yolov8n.pt")

# Apri la webcam (0 = integrata, 1 = USB)
cap = cv2.VideoCapture(0)

# Crea una finestra a tutto schermo
cv2.namedWindow("YOLOv8 Person Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLOv8 Person Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Ottieni dimensioni dello schermo
#screen_width = 1920
#screen_height = 1080

screen_width = 640
screen_height = 480


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Opzionale: crea una versione ridotta per inferenza più veloce
    # small_frame = cv2.resize(frame, (640, 640))
    # results = model(small_frame)[0]
    results = model(frame)[0]  # se vuoi usare risoluzione piena

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

    # Ridimensiona il frame per visualizzarlo a pieno schermo
    full_screen_frame = cv2.resize(frame, (screen_width, screen_height))

    # Mostra l'immagine a tutto schermo
    cv2.imshow("YOLOv8 Person Detection", full_screen_frame)

    # Esci con ESC
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
