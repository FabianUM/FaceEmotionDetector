import cv2
import mediapipe as mp
import math
import tkinter as tk
from tkinter import filedialog

class FaceEmotionDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        
        self.mpDibujo = mp.solutions.drawing_utils
        self.confDibujo = self.mpDibujo.DrawingSpec(thickness=1, circle_radius=1)
        
        self.mpMallaFacial = mp.solutions.face_mesh
        self.mallaFacial = self.mpMallaFacial.FaceMesh(max_num_faces=1)
        
        self.emotions = []

    def detect_emotions(self):
        while True:
            ret, frame = self.cap.read()
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = self.mallaFacial.process(frameRGB)

            px = []
            py = []
            lista = []
            emotion_detected = ""

            if resultados.multi_face_landmarks:
                for rostros in resultados.multi_face_landmarks:
                    self.mpDibujo.draw_landmarks(frame, rostros, self.mpMallaFacial.FACEMESH_CONTOURS, self.confDibujo, self.confDibujo)

                    for id, puntos in enumerate(rostros.landmark):
                        al, an, c = frame.shape
                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)
                        lista.append([id, x, y])

                        if len(lista) == 468:
                            x1, y1 = lista[65][1:]
                            x2, y2 = lista[158][1:]
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            longitud1 = math.hypot(x2 - x1, y2 - y1)

                            x3, y3 = lista[291][1:]
                            x4, y4 = lista[385][1:]
                            longitud2 = math.hypot(x4 - x3, y4 - y3)

                            x5, y5 = lista[78][1:]
                            x6, y6 = lista[308][1:]
                            longitud3 = math.hypot(x6 - x5, y6 - y5)

                            x7, y7 = lista[13][1:]
                            x8, y8 = lista[14][1:]
                            longitud4 = math.hypot(x8 - x7, y8 - y7)

                            if longitud1 < 19 and longitud2 < 19 and longitud3 > 80 and longitud3 < 95 and longitud4 < 5:
                                emotion_detected = 'Angry'
                                cv2.putText(frame, 'Angry', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            elif longitud1 > 30 and longitud2 > 30 and longitud3 > 90 and longitud3 < 110 and longitud4 > 10 and longitud4 < 20:
                                emotion_detected = 'Happy'
                                cv2.putText(frame, 'Happy', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                            elif longitud1 > 35 and longitud2 > 35 and longitud3 > 80 and longitud3 < 95 and longitud4 > 20:
                                emotion_detected = 'Surprised'
                                cv2.putText(frame, 'Surprised', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                            elif longitud1 > 20 and longitud1 < 35 and longitud2 > 25 and longitud2 < 35 and longitud3 > 90 and longitud3 < 95 and longitud4 < 5:
                                emotion_detected = 'Sad'
                                cv2.putText(frame, 'Sad', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            if emotion_detected:
                self.emotions.append(emotion_detected)

            cv2.imshow("Emotion Detection", frame)
            t = cv2.waitKey(1)

            if t == 27:  # ESC para terminar
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.save_results()

    def save_results(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'w') as f:
                for emotion in self.emotions:
                    f.write(f"{emotion}\n")

