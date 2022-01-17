import os
import FaceRecAPI_temp
import cv2

path = '../../new_datasets/'
embedings_base = {}

app = FaceRecAPI_temp.FaceRec()

dirs = os.listdir(path)
for d in dirs:
    files = os.listdir(os.path.join(path + d + '/'))
    for file in files:
        img = cv2.imread(os.path.join(path + d + '/' + file))
        emb = app.get_embeding(img)
        if d not in embedings_base.keys():
            embedings_base.update({d: []})
        embedings_base[d].append(emb)


def find_name(new_img):
    emb_find = app.get_embeding(new_img)
    try:
        for key in embedings_base.keys():
            for item in embedings_base[key]:
                res = app.compare_embedings(emb_find, item)
                if res < 0.4:
                    return key
    except(Exception):
        pass

    return 'XZ'


cap = cv2.VideoCapture(0)
old = ''
while True:
    ret, frame = cap.read()
    if ret:
        res = find_name(frame)
        if old != res:
            old = res
            print(old)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()



