import torch

BATCH_SIZE = 10 # increase / decrease according to GPU memeory
RESIZE_TO = 416 # resize the image for training and transforms
NUM_EPOCHS = 14 # number of epochs to train for
NUM_WORKERS = 4
ROOT = 'C:\\Users\ornst\VLADIMIR_USER\projects\kitty_detection\\'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
TRAIN_DIR = 'data/Uno Cards.v2-raw.voc/train'
TRAIN_IMAGES_DIR = ROOT + 'image-detection-on-kitty-dataset\datasets\kitty\images\\train_65p'
TRAIN_LABELS_DIR = ROOT + 'image-detection-on-kitty-dataset\datasets\kitty\orig_labels\\train_65p_intermediate'
# validation images and XML files directory
VALID_DIR = 'data/Uno Cards.v2-raw.voc/valid'

VALID_IMAGES_DIR = ROOT + 'image-detection-on-kitty-dataset\datasets\kitty\images\\train_25p'
VALID_LABELS_DIR = ROOT + 'image-detection-on-kitty-dataset\datasets\kitty\orig_labels\\train_25p_intermediate'

# classes: 0 index is reserved for background
CLASSES = [str(i) for i in range(9)]

NUM_CLASSES = len(CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = 'outputs'