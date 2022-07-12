import os
from telnetlib import XDISPLOC
import cv2
import random
import numpy as np
import pickle

CATEGORIES = ["happy","nothappy"]
IMG_SIZE = 100
PATH = "data"
training_data = []

def create_dataset():
    for category in CATEGORIES:
        path = os.path.join(PATH,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(array,(IMG_SIZE,IMG_SIZE))
                training_data.append((new_array,class_num))

            except Exception as e:
                print (e)
                pass
create_dataset()
random.shuffle(training_data)
#print(len(training_data))

X, Y = [], []

for features, label in training_data:
    X.append(features)
    Y.append(label)
X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)

#saving X data
pickle_saver = open("data/x.db","wb")
pickle.dump(X,pickle_saver)
pickle_saver.close()

#saving Y data
pickle_saver = open("data/y.db","wb")
pickle.dump(Y,pickle_saver)
pickle_saver.close()

print(X[0].shape)

cv2.imshow("img",X[5])
cv2.waitKey(0)

