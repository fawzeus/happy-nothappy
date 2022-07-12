import pickle
import cv2
import pathlib
import numpy as np
from MODEL import create_model



X = pickle.load(open("data/x.db","rb"))
IMG_SIZE = 100
CATEGORIES = ["happy","not happy"]

model = create_model()
model.load_weights("data/model_weights")


CASCADE_PATH = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(CASCADE_PATH))

camera = cv2.VideoCapture(0)


while True:
    _,frame = camera.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    for x,y,width,height in faces :
        cv2.rectangle(frame,(x,y),(x + width, y +height),(255,255,0),2)
        global face
        face = np.ones((height,width),dtype=np.uint8)*255
        face  = gray[y:y+height,x:x+width]
        #print (width,"   ",height)
    if len(faces):
        face =  np.resize(face,(1,IMG_SIZE,IMG_SIZE,1))
        y_pred = model.predict(face)
        if y_pred[0][0] > 0.5:
            y_pred = 1
        else:
            y_pred = 0
        cv2.putText(frame,f"User is {CATEGORIES[y_pred]}",(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
        #print(f"User is {CATEGORIES[y_pred]}")
    cv2.imshow("live",frame)
    if cv2.waitKey(1) == ord("q"):
        break
