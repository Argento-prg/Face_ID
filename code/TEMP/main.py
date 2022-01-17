

import cv2
import numpy as np
import dlib
import imagehash
from PIL import Image



# Подключение детектора, настроенного на поиск человеческих лиц
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")

# Получение изображения из видео потока

frame = cv2.imread("../images_know/f2.jpg")
# Конвертирование изображения в черно-белое
grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
height, width = grayFrame.shape[:2]
img = np.zeros([height,width,3])# массив нулей, потом будет использоваться как белый фон

img[:,:,0] = np.ones([height,width])
img[:,:,1] = np.ones([height,width])
img[:,:,2] = np.ones([height,width])

cv2.imwrite('color_img.jpg', img)#записываем этот массив в фотку

# Обнаружение лиц и построение прямоугольного контура
faces = detector(grayFrame)

# Обход списка всех лиц попавших на изображение
for face in faces:

    # Выводим количество лиц на изображении
    cv2.putText(frame, "{} face(s) found".format(len(faces)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Получение координат вершин прямоугольника и его построение на изображении
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # Получение координат контрольных точек и их построение на изображении
    landmarks = predictor(frame, face)


    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)#рисуем окружности на исходной фотке
        cv2.circle(img, (x, y), 1, (0, 0, 0), -1)#рисуем окружности на белой фотке
        print(str(x) + " " + str(y))#вывел координаты точек по приколу




# Вывод преобразованного изображения
cv2.imshow("Frame", img)
cv2.waitKey(0)
img = Image.fromarray(np.uint8(img))
hash1 = imagehash.phash(img)
print(hash1)

