import cv2
from scipy.spatial import distance
import math


class FaceRec:
    __dict_dots = {
        0: [8, 17, 36],
        8: [16, 57],
        16: [26, 45],
        17: [19, 21, 36, 48],
        19: [21],
        21: [22, 27, 39],
        22: [24, 26, 27, 42],
        24: [26],
        26: [45, 54],
        27: [30],
        30: [36, 39, 42, 45, 48, 54, 57],
        36: [39],
        39: [42],
        42: [45],
        48: [54, 57],
        54: [57]
    }

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('../../models/haarcascade_frontalface_alt2.xml')
        lbf_model = '../../models/lbfmodel.yaml'
        self.landmark_detector = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel(lbf_model)

    def __get_new_dsize(self, img):
        shape = img.shape
        height = int(shape[0] * 25 / 100)
        width = int(shape[1] * 25 / 100)
        new_dsize = (width, height)
        return new_dsize

    def __get_normal(self, l36, l39, r42, r45):
        xl = (r45[0] + r42[0]) / 2

        yl = (r45[1] + r42[1]) / 2

        xr = (l39[0] + l36[0]) / 2

        yr = (l39[1] + l36[1]) / 2

        normal = self.__calc_len_vector(xl, yl, xr, yr)

        return normal

    def __compute_embeding(self, landmark):
        lst = []
        normal = self.__get_normal(landmark[36], landmark[39], landmark[42], landmark[45])
        for i in self.__dict_dots.keys():
            x1 = landmark[i][0]
            y1 = landmark[i][1]
            for j in self.__dict_dots[i]:
                x2 = landmark[j][0]
                y2 = landmark[j][1]
                lst.append(self.__calc_len_vector(x1, y1, x2, y2) / normal)
        return lst

    def __calc_len_vector(self, x1, y1, x2, y2):
        x = x2 - x1
        y = y2 - y1
        result = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
        return result

    def __process_image(self, img):
        img = cv2.resize(img, dsize=self.__get_new_dsize(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def get_embeding(self, img):
        embedings = None
        try:
            new_img = self.__process_image(img)
            faces = self.face_cascade.detectMultiScale(new_img, scaleFactor=1.1)
            _, landmarks = self.landmark_detector.fit(new_img, faces)
            if _:
                embedings = self.__compute_embeding(landmarks[0][0])
        except(Exception):
            embedings = None

        return embedings

    def compare_embedings(self, emb1, emb2):
        return distance.euclidean(emb1, emb2)


