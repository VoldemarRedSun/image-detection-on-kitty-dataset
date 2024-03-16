from typing import Collection, Sequence

import cv2

def draw_bbox(image, bbox: Sequence, mode: str):

    image = cv2.imread(image)
    size = [0]*2
    size[1], size[0], _ = image.shape
    convert = bbox
    if mode == 'xywhn':
        x1 = int(convert[0] * size[0] - convert[2] * size[0]/2.0)
        y1 = int(convert[1] * size[1] - convert[3] * size[1]/2.0)
        x2 = x1 + int(convert[2] * size[0])
        y2 = y1 + int(convert[3] * size[1])
    if mode == 'xyxy':
        x1, y1, x2, y2 =  map(lambda num: int(num), convert)
    if mode == 'xyxyn':
        x1, x2 = map(lambda num: int(num*size[0]), convert[[0,2]])
        y1, y2 = map(lambda num: int(num*size[1]), convert[[1, 3]])
    # if mode == 'xyxyn_new':
    #     x1, x2 = map(lambda num: int(num*640), convert[[0,2]])
    #     y1, y2 = map(lambda num: int(num*224), convert[[1, 2]])


    print()

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
    cv2.imshow('image', image)
    cv2.waitKey()