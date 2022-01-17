import cv2
import numpy as np
import imagehash
from PIL import Image
import os
from scipy.spatial import distance
# It's try to make face recognition use opencv-python


class AppFace:

    def __init__(self):
        try:
            path_haar = '../models/haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(path_haar)  # путь к Каскадам Хаара
            self.app_work()  # основная функция работы программы
        except Exception as ex:
            # Что то случилось
            print(ex)

    def app_work(self):
        path_to_dir = '../images_know/'
        hashlist1 = []
        hashlist2 = []
        self.count = 0
        fileList = os.listdir(path=path_to_dir)
        for file in fileList:
            frame = cv2.imread(os.path.join(path_to_dir + file))
            print(os.path.join(path_to_dir + file))
            img_processing = self.processing_first(frame)
            img_hash = str(self.process_spec_dots(img_processing))
            hashlist1.append(img_hash)
            self.count += 1

        path_to_dir = '../images_unknow/'
        fileList = os.listdir(path=path_to_dir)
        for file in fileList:
            frame = cv2.imread(os.path.join(path_to_dir + file))
            print(os.path.join(path_to_dir + file))
            img_processing = self.processing_first(frame)
            img_hash = str(self.process_spec_dots(img_processing))
            hashlist2.append(img_hash)
            self.count += 1



        for i in range(len(hashlist1)):
            distance = self.hemming_distance(hashlist1[i], hashlist2[i])
            print('distance = ' + str(distance))
            result = self.result_compare(distance=distance)
            print('result = ' + str(result))




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
        cv2.imwrite('../cut_out/file' + str(self.count) + '.jpg', new_img)
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
        cv2.imwrite('../after_process/file' + str(self.count) + '.jpg', processed_image)
        return processed_image

    def processing_first(self, frame):

        # Первичная обработка изображения
        img = cv2.resize(frame, (640, 480))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #img = self.filter_image(img) # не используем, так как без него результаты лучше
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
        distance1 = 0
        cmp_p_hash = bin(int(p_hash, base=16))
        cmp_etalon = bin(int(etalon, base=16))
        t1 = []
        t2 = []
        for j in range(len(cmp_p_hash)):
            hash1 = str(cmp_p_hash)
            hash2 = str(cmp_etalon)
            if hash1[j] != hash2[j]:
                distance1 += 1
        cmp_etalon = str(cmp_etalon)
        cmp_p_hash = str(cmp_p_hash)
        for j in range(len(cmp_p_hash)):
            t1.append(int(cmp_p_hash[j], base=16))
            t2.append(int(cmp_etalon[j], base=16))
        dst = distance.euclidean(t1, t2)
        return dst

    def result_compare(self, distance):
        result = False
        if distance <= 10:
            result = True
        return result


if __name__ == '__main__':
    App = AppFace()  # начало программы

