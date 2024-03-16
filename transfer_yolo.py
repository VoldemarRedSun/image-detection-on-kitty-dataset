import torch
from yolov5_draw import draw_bbox
if __name__ == '__main__':
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    freeze = 24
    freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False