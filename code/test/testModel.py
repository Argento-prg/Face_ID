import cv2
from scipy.spatial import distance
import math
import os


class FaceRec:
    __dict_dots_v1 = {
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

    __dict_dots_v4 = {
        0: [2, 8, 17, 36],
        2: [8, 31, 36, 48],
        8: [14, 16, 57],
        14: [16, 35, 45, 54],
        16: [26, 45],
        17: [19, 21, 48],
        19: [24, 36, 39],
        21: [22, 27, 39],
        22: [26, 27, 42],
        24: [26, 42, 45],
        26: [54],
        27: [30],
        30: [31, 35, 36, 39, 42, 45, 51],
        31: [35, 39, 48, 50, 51],
        35: [42, 51, 52, 54],
        36: [39, 48],
        39: [42],
        42: [45],
        45: [54],
        48: [51, 54, 57],
        50: [52],
        51: [54, 57],
        54: [57]
    }

    __dict_dots_v5 = {
        0: [1, 17,31,36],
        1: [2, 31],
        2: [3,31],
        3: [4,31],
        4: [5,31,48],
        5: [6, 48],
        6: [7,48],
        7: [8, 58],
        8: [9, 57],
        9: [10, 56],
        10: [11, 54],
        11: [12, 54],
        12: [13,35,54],
        13: [14,35],
        14: [15,35],
        15: [16, 35],
        16: [26,35,45],
        17: [18, 37],
        18: [19],
        19: [20],
        20: [21,37],
        21: [22, 27, 39],
        22: [23, 27, 42],
        23: [24,44],
        24: [25],
        25: [26],
        26: [44],
        27: [28,39,42],
        28: [29],
        29: [30, 39,42],
        30: [31, 32, 33,34,35,39,42],
        31: [32, 39, 50],
        32: [33, 50],
        33: [34,51],
        34: [35, 52],
        35: [42, 52],
        36: [37, 41],
        37: [38],
        38: [39],
        39: [40],
        40: [41],
        42: [43, 47],
        43: [44],
        44: [45],
        45: [46],
        46: [47],
        48: [49, 59,60],
        49: [50,61],
        50: [51, 61],
        51: [52, 62],
        52: [53, 63],
        53: [54,63],
        54: [55,64],
        55: [56,64,65],
        56: [57,65],
        57: [58,66],
        58: [59,67],
        59: [60,67],
        60: [67],
        61: [62,67],
        62: [63,66],
        63: [65],
        64: [65],
        65: [66],
        66: [67]
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
        for i in self.__dict_dots_v4.keys():
            x1 = landmark[i][0]
            y1 = landmark[i][1]
            for j in self.__dict_dots_v4[i]:
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
        #img = cv2.resize(img, dsize=self.__get_new_dsize(img))
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


def get_descriptors(app, path):
    embedings = {}
    dirs = os.listdir(path)
    for dir in dirs:
        files = os.listdir(path + dir + '/')
        for file in files:
            img = cv2.imread(path + dir + '/' + file)
            emb = app.get_embeding(img)
            embedings.update({dir: emb})
    return embedings



if __name__ == "__main__":
    app = FaceRec()
    path = '../../test/bd/'
    bd_emb = get_descriptors(app, path)
    path = '../../test/input/oleg.mp4'
    cap = cv2.VideoCapture(path)
    flag_work = True
    count_right = 0
    count_psevdo = 0
    count_all = 0
    try:
        while flag_work:
            ret, frame = cap.read()
            if ret:
                count_all += 1
                emb = app.get_embeding(frame)
                cv2.imshow('frame', frame)
                for key in bd_emb.keys():
                    test = app.compare_embedings(emb, bd_emb[key])
                    if test < 0.45:
                        if key == 'oleg':
                            count_right += 1
                        else:
                            count_psevdo += 1
                cv2.waitKey(1)
            else:
                flag_work = False
    except(Exception):
        pass
    cap.release()
    cv2.destroyAllWindows()
    print('Right: ', count_right / count_all)
    print('Psevdo: ', count_psevdo / count_all)
    print('Sum: ', (count_psevdo + count_right) / count_all)


