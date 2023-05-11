import numpy as np
import cv2
from TFLiteFaceDetector import UltraLightFaceDetecion
from TFLiteFaceAlignment import CoordinateAlignmentModel

def zoom_effect(_landmarks):
    landmarks = []
    for pred in fa.get_landmarks(image, boxes):
        for i in _landmarks:
            landmarks.append(pred[i])
        landmarks = np.array(landmarks, dtype= int)

    x,y,w,h = cv2.boundingRect(landmarks)
    mask = np.zeros(image.shape, dtype= np.uint8)
    cv2.drawContours(mask,[landmarks],-1,(255,255,255),-1)

    result = image * (mask // 255)
    result = result[y:y+h, x:x+w]
    result = cv2.resize(result,(w*2,h*2))
    ry = int(y-(h/2))
    rx = int(x-(w/2))

    for f in range(2):
        for i in range(h*2):
            for j in range(w*2):
                if result[i][j][f] == 0 and result[i][j][f] == 0 and result[i][j][f] == 0:
                    result[i][j] = image[ry+i, rx+j]

    image[ry:ry+h*2, rx:rx+w*2] = result

    return image

fd = UltraLightFaceDetecion("OpenVtuber/weights/RFB-320.tflite",conf_threshold=0.88)
fa = CoordinateAlignmentModel("OpenVtuber/weights/coor_2d106.tflite")

image = cv2.imread("input/im1.jpg")
boxes, scores = fd.inference(image)

image = zoom_effect([52,64,63,71,67,68,61,58,59,53,56,55])
image = zoom_effect([35,41,40,42,39,37,33,36])
image = zoom_effect([89,95,94,96,93,91,87,90])

cv2.imwrite("outputs_assignment_30/result_2.png", image)