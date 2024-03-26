from functools import partial

import torchvision
from torch import nn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssdlite import SSDLiteHead
from torchvision.models.detection import _utils as det_utils


def create_model(num_classes, model_type: str = 'SSD_lite'):
    # load Faster RCNN pre-trained model
    if model_type == 'Faster_RCNN':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    if model_type == 'SSD_lite':
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
        # in_features = model.backbone.out_channels
        out_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
        # out_channels = model.backbone.out_channels
        num_anchors = model.anchor_generator.num_anchors_per_location()
        # num_anchors = model.backbone.num_anchors
        norm_layer =  partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        model.head = SSDLiteHead(in_channels = out_channels, num_anchors = num_anchors,  num_classes = num_classes, norm_layer = norm_layer)




    return model

if __name__ == '__main__':
    create_model(5, 'SSD_lite')