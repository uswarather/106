import cv2
faceclassifier=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
image=cv2.imread("boy.jpg")
faces=faceclassifier.detectMultiScale(image,1.1,4)
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),6)
cv2.imshow("facedetector",image)
cv2.waitKey()