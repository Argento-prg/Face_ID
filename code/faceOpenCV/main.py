import FaceRecAPI_temp
import cv2
import os

if __name__ == "__main__":
    app = FaceRecAPI_temp.FaceRec()
    path = '../../new_datasets/oleg/'
    files = os.listdir(path)
    count = 0
    emb1 = None
    for file in files:
        img = cv2.imread(os.path.join(path + file))
        emb2 = app.get_embeding(img)
        if count == 0:
            count += 1
            emb1 = emb2
        else:
            print(app.compare_embedings(emb1, emb2))
print('camera')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        emb2 = app.get_embeding(frame)
        print(app.compare_embedings(emb1, emb2))
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()