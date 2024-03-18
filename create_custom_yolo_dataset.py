from typing import Dict
import yaml
from PIL import Image
import os
from os.path import isfile, join

from draw_utils import convertToYoloBBox


def convert_txt_kitty2yolo(from_path: str,
                           ind2name_dict: Dict = yaml.safe_load(open('yolov5/data/kitty.yaml'))['names']) -> str:
    entire_annotation = ''
    with open(from_path) as file:
        lines = [line.rstrip() for line in file]
    name2ind_dict = dict(zip(ind2name_dict.values(),ind2name_dict.keys()))
    for line in lines:
        processed_line = str(name2ind_dict[line.split(' ')[0]]) +' ' + ' '.join(line.split(' ')[4:8])
        entire_annotation += processed_line + '\n'
    entire_annotation
    return entire_annotation


def create_yolo_label_dir(source_dir: str = 'datasets\kitty\orig_labels\\train_test',
                          target_dir: str = 'datasets\kitty\labels\\train_test_new_format'):
    annotations_files = os.listdir(source_dir)
    annotations_files = list(map(lambda file: os.path.join(source_dir, file), annotations_files))
    for i, file_path in enumerate(annotations_files):
        yolo_annot = convert_txt_kitty2yolo(file_path)
        postfix = file_path.split('\\')[-1]
        with open(target_dir + f'\\{postfix}', 'w') as target_file:
            target_file.write(yolo_annot)


def create_normalize_label_dir(source_annot_dir: str = 'datasets\kitty\labels\\train_test_old_format',
                               target_annot_dir: str = 'datasets\kitty\labels\\train_test',
                               images_dir: str = 'datasets\kitty\images\\train_test'):

    annotations_files = os.listdir(source_annot_dir)
    annotations_files = list(map(lambda file: os.path.join(source_annot_dir, file), annotations_files))
    image_paths, image_shapes = get_images_shape()

    for annot_path, img_size in zip(annotations_files, image_shapes):
        with open(annot_path) as file:
            lines = [line.rstrip() for line in file]
            classes = []
            bboxes = []
            new_annot = ''
            for line in lines:
                ind, x1, y1, x2, y2 = line.split(' ')
                bbox = convertToYoloBBox([float(x1), float(y1), float(x2), float(y2)], img_size)
                bboxes.append(bbox)
                classes.append(int(line[0]))
                new_annot += f'{ind} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n'

        postfix = annot_path.split('\\')[-1]
        with open(target_annot_dir + f'\\{postfix}', 'w') as target_file:
            target_file.write(new_annot)


def get_images_shape(images_dir: str = 'datasets\kitty\images\\train_test'):
    image_paths = os.listdir(images_dir)
    image_paths = list(map(lambda file: os.path.join(images_dir, file), image_paths))
    img_sizes = [Image.open(path).size for path in image_paths]
    return image_paths, img_sizes

if __name__ == '__main__':
    # kitty_labels_dir = 'datasets\original_kitty\labels'
    # annotations_files = os.listdir(kitty_labels_dir)
    # with open('yolov5/data/kitty.yaml') as f:
    # config = yaml.safe_load(f)
    # create_yolo_label_dir()
    # get_images_shape()
    create_normalize_label_dir()
