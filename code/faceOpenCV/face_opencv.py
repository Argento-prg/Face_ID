import cv2
import numpy as np
import imagehash
from PIL import Image
from scipy.spatial import distance
# It's try to make face recognition use opencv-python


class AppFace:

    def __init__(self):
        try:
            self.face_cascade = cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')  # путь к Каскадам Хаара
            self.app_work()  # основная функция работы программы
        except Exception as ex:
            # Что то случилось
            print(ex)

    def app_work(self):
        print("Press \'q\' to exit")
        cap = cv2.VideoCapture(0)  # Инициализация соединения с веб-камерой

        flag_work = True

        while flag_work:
            ret, frame = cap.read()
            if ret:
                img_processing = self.processing_first(frame)
                print(self.process_spec_dots(img_processing))# пока происходит вывод хеша дескриптора изображения в консоль
                cv2.imshow('frame', img_processing)

                if cv2.waitKey(1) == ord('q'):
                    flag_work = False
                    cap.release()
                    cv2.destroyAllWindows()
                    print("End of program")
            else:
                cap.release()
                print("NO FRAME!")

    def cut_face(self, img):
        # Вырезаем лицо из картинки
        faces = self.face_cascade.detectMultiScale(img, scaleFactor=1.1)
        new_img = img

        if len(faces) > 0:
            x = faces[0][0]
            y = faces[0][1]
            x_with_len = x + faces[0][2]
            y_with_len = y + faces[0][3]
            new_img = img[y: y_with_len, x:x_with_len]

        return new_img

    def filter_image(self, img):
        # делаем изображение чётче
        kernel = [-0.1, -0.1, -0.1, -0.1, 2.0, -0.1, -0.1, -0.1, -0.1]
        kernel = np.array(kernel)
        img_new = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
        return img_new

    def delete_noise(self, img):
        # медианная фильтрация
        processed_image = cv2.medianBlur(img, 3)
        return processed_image

    def processing_first(self, frame):

        # Первичная обработка изображения
        img = cv2.resize(frame, (640, 480))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # img = self.filter_image(img) # не используем, так как без него результаты лучше
        img = self.delete_noise(img)
        img = self.cut_face(img)
        return img

    def process_spec_dots(self, frame):
        # find special dots
        orb = cv2.ORB_create()
        kp = orb.detect(frame)
        kp, des = orb.compute(frame, kp)
        result_hash = str(imagehash.phash(Image.fromarray(des)))
        return result_hash

    def hemming_distance(self, p_hash, etalon):
        # считаем хеммингово расстояние между двумя хешами
        distance = 0
        cmp_p_hash = str(bin(int(p_hash, base=16)))
        cmp_etalon = str(bin(int(etalon, base=16)))
        for j in range(len(cmp_p_hash)):
            if cmp_etalon[j] != cmp_p_hash[j]:
                distance += 1
        return distance

    def euclidean_distance(self, p_hash, etalon):
        # Евклидово расстояние
        cmp_p_hash = str(bin(int(p_hash, base=16)))
        cmp_etalon = str(bin(int(etalon, base=16)))
        t1 = []
        t2 = []
        for j in range(len(cmp_p_hash)):
            t1.append(int(cmp_p_hash[j], base=16))
            t2.append(int(cmp_etalon[j], base=16))
        dst = distance.euclidean(t1, t2)
        return dst

    def result_compare(self, distance):
        # пока не используется
        result = False
        if distance <= 10:
            result = True
        return result


if __name__ == '__main__':
    App = AppFace()  # начало программы

