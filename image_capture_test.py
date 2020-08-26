import cv2
import os
from pathlib import Path

img_counter = 0
#for counter in range(555):
#    fname = "opencv_frame_{}.png".format(counter)

#    print(Path(fname))
#    if Path(fname).is_file():
#        print ("File exist")
#        print(fname)
#        continue
#    else:
#        print ("File not exist")
#        img_counter = counter
#        print(img_counter)
#        break




cam = cv2.VideoCapture(1)

#cv2.namedWindow("test")

while True:
    ret, frame = cam.read()
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow('object detection', frame)
    while True:
        if cv2.waitKey(20) == ord('q'):
                img_name = "opencv_frame_{}.png".format(2)
                cv2.imwrite(img_name, frame)
                break

    #cv2.imshow("test", gray)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    #img_name = "opencv_frame_{}.png".format(img_counter)
    #cv2.imwrite(img_name, frame)
    #print("{} written!".format(img_name))
    img_counter += 1

cam.release()

cv2.destroyAllWindows()