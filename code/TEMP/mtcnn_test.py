import cv2
from mtcnn import MTCNN
import datetime

print(datetime.datetime.now())
detector = MTCNN()
image = cv2.cvtColor(cv2.imread("../images_know/papa1.jpg"), cv2.COLOR_BGR2RGB)
result = detector.detect_faces(image)
bounding_box = result[0]['box']
keypoints = result[0]['keypoints']
cv2.rectangle(image,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0, 155, 255), 2)
cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

print(datetime.datetime.now())

cv2.imwrite("ivan_drawn.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
print(result)
