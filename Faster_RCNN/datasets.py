import torch
import cv2
import numpy as np
import os



from config import (
    CLASSES, RESIZE_TO, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, VALID_IMAGES_DIR, VALID_LABELS_DIR, BATCH_SIZE
)
from torch.utils.data import Dataset, DataLoader
from custom_utils import collate_fn, get_train_transform, get_valid_transform

# the dataset class
class CustomDataset(Dataset):
    def __init__(self, images_dir_path, labels_dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.images_dir_path = images_dir_path
        self.labels_dir_path = labels_dir_path

        self.height = height
        self.width = width
        self.classes = classes

        images_paths = os.listdir(images_dir_path)
        labels_paths = os.listdir(labels_dir_path)
        self.all_images = list(map(lambda file: os.path.join(images_dir_path, file), images_paths))
        self.all_labels = list(map(lambda file: os.path.join(labels_dir_path, file), labels_paths))
    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_path = self.all_images[idx]
        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        
        # capture the corresponding XML file for getting the annotations
        annot_file_path = self.all_labels[idx]
        
        boxes = []
        labels = []

        
        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]


        with open(annot_file_path) as file:
            lines = [line.rstrip() for line in file]
            classes = []
            bboxes = []
            for line in lines:
                ind, x1, y1, x2, y2 = line.split(' ')
                xmin = float(x1)
                xmax = float(x2)
                ymin = float(y1)
                ymax = float(y2)


                xmin_final = (xmin/image_width)*self.width
                xmax_final = (xmax/image_width)*self.width
                ymin_final = (ymin/image_height)*self.height
                ymax_final = (ymax/image_height)*self.height
            
                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
                labels.append(int(ind))
        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
        return image_resized, target

    def __len__(self):
        return len(self.all_images)

# prepare the final datasets and data loaders
def create_train_dataset():
    train_dataset = CustomDataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
    return train_dataset
def create_valid_dataset():
    valid_dataset = CustomDataset(VALID_IMAGES_DIR, VALID_LABELS_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
    return valid_dataset

def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader
def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader


# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    dataset = CustomDataset(
        TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    print(f"Number of training images: {len(dataset)}")
    
    # function to visualize a single sample
    def visualize_sample(image, target):
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 2
            )
            cv2.putText(
                image, str(label), (int(box[0]), int(box[1]-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        
    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)