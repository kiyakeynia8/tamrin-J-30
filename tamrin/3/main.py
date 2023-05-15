import numpy as np
import cv2
from TFLiteFaceDetector import UltraLightFaceDetecion
from TFLiteFaceAlignment import CoordinateAlignmentModel
import PIL
import math

img = cv2.imread("input\im4.png") 

fd = UltraLightFaceDetecion("OpenVtuber\weights\RFB-320.tflite")
fa = CoordinateAlignmentModel("OpenVtuber\weights\coor_2d106.tflite")

boxes, scores = fd.inference(img)
lock_landmarks = []

for pred in fa.get_landmarks(img , boxes):
    ...

for landmark in [38,88,55,61,84]:
    lock_landmarks.append(pred[landmark])

left_eye = lock_landmarks[0]
right_eye = lock_landmarks[1]
left_eye_x , left_eye_y  = left_eye
right_eye_x , right_eye_y  = right_eye

direction = 1
point_3rd = (left_eye_x, right_eye_y)

if left_eye_y > right_eye_y:
    point_3rd = (right_eye_x, left_eye_y)
    direction = -1

def Distance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    
    return euclidean_distance

a = Distance(left_eye, point_3rd)
b = Distance(right_eye, point_3rd)
c = Distance(right_eye, left_eye)

cos_a = math.cos(b)
angle = np.arccos(cos_a)
angle = (angle * 180) / 3.1415

angle = 90 - angle

img = PIL.Image.fromarray(img)
img = np.array(img.rotate(direction * angle))

cv2.imshow("result", img)
cv2.waitKey()
cv2.imwrite("outputs_assignment_30/f_result_3.png",img)