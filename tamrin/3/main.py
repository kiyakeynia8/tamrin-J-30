from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
# import math

def draw_facebox_and_keypoints(filename, result_list):
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()
    for result in result_list:
        x, y, width, height = result['box']
        rect = plt.Rectangle((x, y), width, height,fill=False, color='orange')
        ax.add_patch(rect)
        for key, value in result['keypoints'].items():
            dot = plt.Circle(value, radius=2, color='red')
            ax.add_patch(dot)
    plt.show()

img = plt.imread("input/im4.png")

face_detector = MTCNN()

results = face_detector.detect_faces(img)

draw_facebox_and_keypoints('input/im4.png', results)