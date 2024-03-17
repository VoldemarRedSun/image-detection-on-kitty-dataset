from typing import Dict
import yaml

import os
from os.path import isfile, join


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
                          target_dir: str = 'datasets\kitty\labels\\train_test'):
    annotations_files = os.listdir(source_dir)
    annotations_files  = list(map(lambda file: os.path.join(source_dir, file), annotations_files))
    for i, file_path in enumerate(annotations_files):
        yolo_annot = convert_txt_kitty2yolo(file_path)
        postfix = file_path.split('\\')[-1]
        with open(target_dir + f'\\{postfix}', 'w') as target_file:
            target_file.write(yolo_annot)



if __name__ == '__main__':
    # kitty_labels_dir = 'datasets\original_kitty\labels'
    # annotations_files = os.listdir(kitty_labels_dir)
    # with open('yolov5/data/kitty.yaml') as f:
    # config = yaml.safe_load(f)
    create_yolo_label_dir()
