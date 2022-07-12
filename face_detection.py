import cv2
import pathlib
import numpy as np

from cv2 import waitKey
#nb = 0

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
    cv2.imshow("live",frame)
    """if len(faces):
        cv2.imshow("face",face)
        cv2.imwrite(f"data/happy/{nb}.jpg",face)
        nb+=1
        print (f"{nb} images saved")"""
    if waitKey(1) == ord("q"):
        break