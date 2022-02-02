
import cv2
import numpy as np
import dlib

webcam = True
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread('test1.jpg')
    img = cv2.resize(img, (0, 0), None, 1, 1)
    imgOriginal = img.copy()

    faces = detector(imgOriginal)
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        imgOriginal=cv2.rectangle(imgOriginal, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #landmarks = predictor(imgGray, face)
        landmarks = predictor(imgOriginal, face)
        myPoints = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x, y])

        def createBox(img, points, scale=5, masked= False,cropped= True):
            if masked:
                mask =  np.ones_like(img)    #   np.zeros_like(img)
                mask = cv2.fillPoly(mask, [points], (255, 255, 255))
                img = cv2.bitwise_and(img, mask)
                #cv2.imshow('Mask',mask)
            if cropped:
                bbox = cv2.boundingRect(points)
                x, y, w, h = bbox
                imgCrop = img[y:y + h, x:x + w]
                imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
                #cv2.imwrite("Mask.jpg", imgCrop)
                return imgCrop
            else:
                return mask
        if len(myPoints) != 0:
            try:
                myPoints = np.array(myPoints)

                imgLeftEye = createBox(img, myPoints[36:42])
                imgRightEye = createBox(img, myPoints[42:48])
                img_Lips = createBox(img, myPoints[48:61])

                maskLeftEye = createBox(img, myPoints[36:42], masked=True, cropped=False)
                maskRightEye = createBox(img, myPoints[42:48], masked=True, cropped=False)
                mask_Lips = createBox(img, myPoints[48:61], masked=True, cropped=False)
                imgColorLeftEye = np.zeros_like(maskLeftEye)
                imgColorRightEye = np.zeros_like(maskRightEye)
                img_ColorLips = np.zeros_like(mask_Lips)


                imgColorLeftEye[:] = 255,255,255  #white color
                imgColorLeftEye = cv2.bitwise_and(maskLeftEye,imgColorLeftEye)     # imgColorLips
                imgColorLeftEye = cv2.GaussianBlur(imgColorLeftEye, (7, 7), 10)
                imgColorLeftEye = cv2.addWeighted(imgOriginal, 1, imgColorLeftEye, 0.4, 0)

                imgColorRightEye[:] = 255,255,255  # white color
                imgColorRightEye = cv2.bitwise_and(maskRightEye, imgColorRightEye)  # imgColorLips
                imgColorRightEye = cv2.GaussianBlur(imgColorRightEye, (7, 7), 10)
                imgColorRightEye = cv2.addWeighted(imgColorLeftEye, 1, imgColorRightEye, 0.4, 0)

                img_ColorLips[:] = 255,255,255  # white color
                img_ColorLips = cv2.bitwise_and(mask_Lips, img_ColorLips)  # imgColorLips
                img_ColorLips = cv2.GaussianBlur(img_ColorLips, (7, 7), 10)
                img_ColorLips = cv2.addWeighted(imgColorRightEye, 1, img_ColorLips, 0.4, 0)
            except:
                pass

        cv2.imshow("Originial",img_ColorLips)
        #print(img_ColorLips.size)
    key=cv2.waitKey(1)
    if key == 27:
        break
print(img_ColorLips.size)
cropped_img = img_ColorLips[y1:y2, x1:x2]
img_resize = cv2.resize(cropped_img, (0,0), None, 5,5)
#imgfinal = cv2.cvtColor(img_resize, cv2.COLOR_GRAY2BGR)
cv2.imwrite('cropped_pic/final image.jpg', img_resize)
