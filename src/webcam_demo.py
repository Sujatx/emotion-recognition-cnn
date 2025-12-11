# src/webcam_demo.py
import cv2
import torch
import torch.nn.functional as F
from .model import EmotionCNN
from .dataset import EMOTION_MAP
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_face(face_gray):
    # face_gray is a 2D numpy array (grayscale)
    face_resized = cv2.resize(face_gray, (48, 48))
    img = face_resized.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)   # channel dim
    img = np.expand_dims(img, axis=0)   # batch dim
    tensor = torch.from_numpy(img).to(DEVICE)
    return tensor

def main():
    model = EmotionCNN(num_classes=7).to(DEVICE)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    model.eval()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)  # default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FIX: flip first so everything (detection + drawing) uses the mirrored-correct frame
        frame = cv2.flip(frame, 1)

        # operate on the flipped frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # expand box slightly
            margin = int(0.2 * w)
            x0 = max(0, x - margin)
            y0 = max(0, y - margin)
            x1 = min(frame.shape[1], x + w + margin)
            y1 = min(frame.shape[0], y + h + margin)

            face = gray[y0:y1, x0:x1]

            # ensure non-empty crop
            if face.size == 0:
                continue

            inp = preprocess_face(face)
            with torch.no_grad():
                logits = model(inp)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(probs.argmax())
                label = EMOTION_MAP[pred_idx]
                conf = probs[pred_idx]

            # draw box + label on the flipped frame (so text is readable)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0,255,0), 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Emotion Demo - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
