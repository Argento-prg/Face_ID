from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial import distance
import numpy as np


# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN()

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='casia-webface').eval()
path = '../../new_datasets/oleg/f0.jpg'
img = Image.open(path)

# Get cropped and prewhitened image tensor
img_cropped = mtcnn(img)

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding = resnet(img_cropped.unsqueeze(0))

path = '../../new_datasets/oleg/f1.jpg'
img = Image.open(path)

# Get cropped and prewhitened image tensor
img_cropped = mtcnn(img)

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding_1 = resnet(img_cropped.unsqueeze(0))
print(distance.euclidean(np.ndarray(img_embedding), np.ndarray(img_embedding_1)))