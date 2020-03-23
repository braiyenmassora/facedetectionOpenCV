
#import library 
import numpy as np
import cv2

#import cascade on your directory
face = cv2.CascadeClassifier('/Users/braiyenmassora/Desktop/openCVfaceDetection/cascades/haarcascade_frontalface_default.xml')
# this function to open camera, if you use external camera you can change into (1)

cap = cv2.VideoCapture(0)

#set width and heigth
cap.set(3,640)
cap.set(4,480)

#looping and detection face 
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(
        gray,1.3,5)


#drawing rectangle in face
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0, 0, 0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        #draw label in face
        cv2.putText(img, 'face', (x-10,y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 0), 2,)
        
#display image in frame video
    cv2.imshow('video',img)

#function to close windows # press 'ESC' to quit
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break

cap.release()
cv2.destroyAllWindows()
