import pickle
import cv2

pickle_reader = open("data/x.db","rb")
X = pickle.load(pickle_reader)


for img in X :
    cv2.imshow("img",img)
    cv2.waitKey(1000)