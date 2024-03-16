def convertToYoloBBox(bbox, size):
# Yolo uses bounding bbox coordinates and size relative to the image size.
# This is taken from https://pjreddie.com/media/files/voc_label.py .
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (bbox[0] + bbox[2]) / 2.0
    y = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


if __name__ == '__main__':
    import cv2

    image = cv2.imread('000000.png')
    x1, y1, x2, y2 = map(lambda num: int(num), [712.40, 143.00, 810.73, 307.92])
    convert =convertToYoloBBox([x1, y1, x2, y2], image.shape[:2])

    x1 = int(convert[0] * image.shape[0] - convert[2] * image.shape[0]/2.0)
    y1 = int(convert[1] * image.shape[1] - convert[3] * image.shape[1]/2.0)
    x2 = x1 + int(convert[2] * image.shape[0])
    y2 = y1 + int(convert[3] * image.shape[1])

    print()

    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0))
    cv2.imshow('pedestrian', image)
    cv2.waitKey()