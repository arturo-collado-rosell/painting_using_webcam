import cv2
import numpy as np

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop



faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#anonimus_face = cv2.imread('anonimus.png')
# cv2.imshow('anonimus',anonimus_face) 
# cv2.waitKey(100) 

frame_Width = 480
frame_Height = 720
captura = cv2.VideoCapture(0)
captura.set(3, frame_Width)
captura.set(4, frame_Height)
while (captura.isOpened()):
    ret, imagen = captura.read()
    if ret == True:
        imagen = cv2.resize(imagen, (720, 480))
        imgGray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('video', imgGray)
        faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
        x,y = 0,0
        for (x,y,w,h) in faces:
            pass
            #cv2.rectangle(imagen, (x,y), (x+w, y+h), (255,0,0), 2)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break    
        # 
        # cv2.imshow('video', imagen)
# 		
		
        # cv2.imshow('video', imagen)
        #alpha_mask = anonimus_face[:, :, 2] / 255.0
        #img_result = imagen[:, :, :3].copy()
        #img_overlay = anonimus_face[:, :, :3]
        #overlay_image_alpha(img_result, img_overlay, x, y, alpha_mask)
        #cv2.imshow('Result', img_result)
        cv2.imshow('Result', imagen)
        
# if cv2.waitKey(1) & 0xFF == ord('s'):
        
    else: break
captura.release()
cv2.destroyAllWindows()
