import cv2
import datetime
import numpy as np
from scipy.spatial import distance
from PIL import Image
import os
import math



dict_dots = {
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

dict_dots_v2 = {
    2: [31, 36, 48],
    14: [35, 45, 54],
    19: [24, 36, 39],
    24: [42, 45],
    30: [31, 35, 39, 42],
    31: [39, 48, 51],
    35: [42, 51, 54],
    39: [42],
    48: [51],
    51: [54]
}





face_cascade = cv2.CascadeClassifier('../../models/haarcascade_frontalface_alt2.xml')
LBF_model = '../../models/lbfmodel.yaml'

landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBF_model)


path_images_dir = '../../datasets/ppl/man_7/'
files = os.listdir(path=path_images_dir)


def get_new_dsize(img):
    shape = img.shape
    height = int(shape[0] * 25 / 100)
    width = int(shape[1] * 25 / 100)
    new_dsize = (width, height)
    return new_dsize


def get_normal(l36, l39, r42, r45):

    xl = (r45[0] + r42[0]) / 2

    yl = (r45[1] + r42[1]) / 2

    xr = (l39[0] + l36[0]) / 2

    yr = (l39[1] + l36[1]) / 2

    dx = xr - xl
    dy = yr - yl

    normal = math.sqrt(dx**2 + dy**2)

    return normal


def compute_embeding(landmark):
    lst = []
    normal = get_normal(landmark[36], landmark[39], landmark[42], landmark[45])
    for i in dict_dots.keys():
        x1 = landmark[i][0]
        y1 = landmark[i][1]
        for j in dict_dots[i]:
            x2 = landmark[j][0]
            y2 = landmark[j][1]
            lst.append(calc_len_vector(x1, y1, x2, y2) / normal)
    return lst


def calc_len_vector(x1, y1, x2, y2):
    x = x2 - x1
    y = y2 - y1
    result = math.sqrt(x**2 + y**2)
    return result



flag = True
embedings = None
new_embedings = None
count = 0
sum_times = 0
for file in files:
    try:
        start = datetime.datetime.now()
        img = cv2.imread(os.path.join(path_images_dir + file))
        img = cv2.resize(img, dsize=get_new_dsize(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1)
        _, landmarks = landmark_detector.fit(img, faces)
        for landmark in landmarks:
            print(file)
            embedings = compute_embeding(landmark[0])
            if flag:
                flag = False
                new_embedings = embedings
            else:
                print(distance.euclidean(embedings, new_embedings))

        end = datetime.datetime.now()
        if count == 0:
            sum_times = end - start
        else:
            sum_times += (end - start)
        count += 1
    except(Exception):
        pass


print(sum_times/count)
print(count)
