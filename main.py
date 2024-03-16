with open("000000.txt") as file:
    mask  = file.readline()
print(mask)
bbox = mask[4:8]
x1,y1,x2,y2 = map(lambda num: int(num), [712.40 , 143.00 , 810.73,  307.92])
# x,y,w,h = [712.40 , 143.00 , 810.73,  307.92])
# print(x,y,w,h)
import cv2

image = cv2.imread('000000.png')
copy = image.copy()



ROI_number = 0



color = (255, 0, 0)
cv2.rectangle(copy,(x1, y1), (x2, y2), color = color)
ROI_number += 1


cv2.imshow('copy', copy)
cv2.waitKey()
