import torch

BATCH_SIZE = 10
RESIZE_TO = 416
NUM_EPOCHS = 8
NUM_WORKERS = 6
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
CLASSES = [str(i) for i in range(89)]

NUM_CLASSES = len(CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = 'outputs'