import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mpDibujo = mp.solutions.drawing_utils
confDibujo = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)

mpMallaFacial = mp.solutions.face_mesh
mallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resultados = mallaFacial.process(frameRGB)

    px = []
    py = []
    lista = []
    r = 5
    t = 3

    if resultados.multi_face_landmarks:
        for rostros in resultados.multi_face_landmarks:
            mpDibujo.draw_landmarks(frame, rostros, mpMallaFacial.FACEMESH_CONTOURS, confDibujo, confDibujo)

            for id, puntos in enumerate(rostros.landmark):
                al, an, c = frame.shape
                x,y = int(puntos.x * an), int(puntos.y * al)
                px.append(x)
                py.append(y)
                lista.append([id,x,y])
                if len(lista) == 468:
                    x1, y1 = lista[65][1:]
                    x2, y2 = lista[158][1:]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    longitud1 = math.hypot(x2 - x1, y2 - y1)

                    x3, y3 = lista[291][1::]
                    x4, y4 = lista[385][1::]
                    cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                    longitud2 = math.hypot(x4 - x3, y4 - y3)

                    x5, y5 = lista[78][1::]
                    x6, y6 = lista[308][1::]
                    cx3, cy3 = (x5 + x6) // 2, (y5 + y6) // 2
                    longitud3 = math.hypot(x6 - x5, y6 - y5)

                    x7, y7 = lista[13][1::]
                    x8, y8 = lista[14][1::]
                    cx4, cy4 = (x7 + x8) // 2, (y7 + y8) // 2
                    longitud4 = math.hypot(x8 - x7, y8 - y7)

                    if longitud1 < 19 and longitud2 < 19 and longitud3 > 80 and longitud3 < 95 and longitud4 < 5:
                        cv2.putText(frame, 'Persona Enojada', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        print("hola")
                    elif longitud1 > 30 and longitud2 > 30 and longitud3 > 90 and longitud3 < 110 and longitud4 > 10 and longitud4 < 20:
                        cv2.putText(frame, 'Persona Feliz', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                        print("hola2")
                    elif longitud1 > 35 and longitud2 > 35 and longitud3 > 80 and longitud3 < 95 and longitud4 > 20:
                        cv2.putText(frame, 'Persona Asombrada', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        print("hola3")
                    elif longitud1 > 20 and longitud1 < 35 and longitud2 > 25 and longitud2 < 35 and longitud3 > 90 and longitud3 < 95 and longitud4 < 5:
                        cv2.putText(frame, 'Persona Triste', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                        print("hola4")


    cv2.imshow("reconocimiento de emociones", frame)
    t = cv2.waitKey(1)

    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()