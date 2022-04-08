import cv2
import numpy as np

frame_Width = 480
frame_Height = 720
captura = cv2.VideoCapture(0)
captura.set(10,120)


myColors = [[0,162,219,179,240,255],
            [1,162,219,179,240,255]]

def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area>300:
            cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x, y, w, h = cv2.boundingRect(approx)            

def findColor(img, myColors):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        getContours(mask)
        #cv2.imshow(str(color[0]),mask)



while (captura.isOpened()):
    ret, imagen = captura.read()
    imgResult = imagen.copy()
    if ret == True:
        imagen = cv2.resize(imagen, (frame_Height, frame_Width))
        findColor(imagen, myColors)
        cv2.imshow('video', imgResult)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
        
        
        
    else: break
captura.release()
cv2.destroyAllWindows()