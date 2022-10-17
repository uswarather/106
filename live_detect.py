import cv2
faseclassifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyedectector=cv2.CascadeClassifier("haarcascade_eye.xml")
camera=cv2.VideoCapture(0)
while True:
    ret,frame=camera.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faseclassifier.detectMultiScale(frame,1.1,4)
    for (x,y,w,h) in faces:
     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),6)
    eye=eyedectector.detectMultiScale(frame,1.1,4)
    for (x,y,w,h) in eye:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),6)
    cv2.imshow("facedetector",frame)
    if cv2.waitKey(1)==32:
        break
