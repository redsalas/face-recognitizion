import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
eyeglasses_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while(True):
    #frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+h]
        roi_color = frame[y:y+h, x:x+h]
        #img_item = "img.png"
        #cv2.imwrite(img_item, roi_gray)

        color = (0,255,0) #BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)

    #Display frame
    cv2.imshow('Capture', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#Release the capture
cap.release()
cv2.destroyAllWindows()

