import cv2
import imutils
import time
import tkinter as tk
from tensorflow.keras.models import load_model
import numpy as np

# Cargamos los modelos de detección de rostros y emociones
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
emotionModel = load_model("modelFEC.h5")
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Ventana con botón para detener el escaneo
def create_stop_window():
    stop = tk.Tk()
    stop.title("Detener escaneo")
    
    stop_flag = tk.BooleanVar()
    stop_flag.set(False)

    def stop_program():
        stop_flag.set(True)
        stop.destroy()

    stop_button = tk.Button(stop, text="Detener", command=stop_program, height=2, width=20)
    stop_button.pack(pady=20)

    stop.mainloop()
    return stop_flag.get()

# Función para guardar las emociones en un archivo
def save_emotions(emotions):
    with open("detected_emotions.txt", "w") as f:
        for emotion in emotions:
            f.write(f"{emotion}\n")

def run_emotion_detection():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    emotions_detected = []  # Para guardar emociones detectadas

    while True:
        ret, frame = cam.read()
        frame = imutils.resize(frame, width=640)

        (locs, preds) = predict_emotion(frame, faceNet, emotionModel)
        for (box, pred) in zip(locs, preds):
            (Xi, Yi, Xf, Yf) = box
            emotion_label = classes[np.argmax(pred)]
            emotions_detected.append(emotion_label)  # Guardar la emoción detectada
            label = "{}: {:.0f}%".format(emotion_label, max(pred) * 100)
            cv2.rectangle(frame, (Xi, Yi-40), (Xf, Yi), (255,0,0), -1)
            cv2.putText(frame, label, (Xi+5, Yi-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255,0,0), 3)

        cv2.imshow("Frame", frame)

        # Comprobar si se ha detenido el programa
        if create_stop_window():
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    save_emotions(emotions_detected)  # Guardar las emociones detectadas en un archivo

def predict_emotion(frame, faceNet, emotionModel):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces, locs, preds = [], [], []

    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")
            face = frame[Yi:Yf, Xi:Xf]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face2 = np.expand_dims(np.array(face), axis=0)
            pred = emotionModel.predict(face2)
            locs.append((Xi, Yi, Xf, Yf))
            preds.append(pred[0])

    return (locs, preds)
