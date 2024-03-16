import torch
from yolov5_draw import draw_bbox
if __name__ == '__main__':
    # Model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

    # Images
    img = "000000.png"  # or file, Path, PIL, OpenCV, numpy, list

    # Inference
    results = model(img)

    # results.show()
    # draw_bbox(img, results.xywhn[0][0][:4])
    draw_bbox(img, results.xyxyn[0][0][:4], mode = 'xyxyn')
    # Results
    results