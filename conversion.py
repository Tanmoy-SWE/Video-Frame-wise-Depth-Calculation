import cv2
import os

cam = cv2.VideoCapture("Pen_Video.mov")

try :
    if not os.path.exists('image_depth_data'):
        os.makedirs('image_depth_data')
except OSError:
    print("The Depth image data directory does not exist.")

currentframe = 0

while(True):
    ret, frame = cam.read()
    if ret :
        name = 'image_depth_data/image_depth_data' + str(currentframe) + '.jpg'
        print("Creating..." +name)

        cv2.imwrite(name, frame)

        currentframe += 1
    else:
        break

cam.release()