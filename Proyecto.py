import cv2 #captura el video desde la cámara.
import mediapipe as mp #detecta la mano y sus puntos clave.

# Inicializar MediaPipe y OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Inicializamos la cámara
cap = cv2.VideoCapture(0) 

# Clase para el objeto virtual
class DraggableObject:
    def __init__(self, pos, size=60, color=(255, 0, 255)):
        self.pos = pos
        self.size = size
        self.color = color
        self.selected = False

    def update(self, cursor):
        cx, cy = self.pos
        cursor_x, cursor_y = cursor

        # Verificar si el cursor está dentro del objeto
        if cx - self.size < cursor_x < cx + self.size and cy - self.size < cursor_y < cy + self.size:
            self.selected = True
            self.pos = cursor
        else:
            self.selected = False

    def draw(self, img):
        cx, cy = self.pos
        cv2.rectangle(img, (cx - self.size, cy - self.size),
                      (cx + self.size, cy + self.size), self.color, -1)

# Crear un objeto virtual
box = DraggableObject(pos=(300, 300))

def detectar_agarre(lmList):
    # Índice y pulgar
    x1, y1 = lmList[4][1:]  # Pulgar
    x2, y2 = lmList[8][1:]  # Índice
    distancia = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    return distancia < 40  # umbral de "agarre"

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # espejo
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            h, w, _ = img.shape
            for id, lm in enumerate(handLms.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                lmList.append([id, px, py])

            if lmList:
                cursor = (lmList[8][1], lmList[8][2])  # dedo índice
                if detectar_agarre(lmList):
                    box.update(cursor)
                box.draw(img)

            mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Arrastrar y Soltar - Seguimiento de Manos", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
