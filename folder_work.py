from pathlib import Path
import shutil
import os

def create_images_folder():
    p = 0.65
    src = 'datasets\kitty\images\\train'
    trg = 'datasets\kitty\images\\train_65p'

    files = os.listdir(src)
    num_files = len(files)
    files_p = files[:int(num_files*0.65)]
    # print(files_25p)

    for fname in files_p:
        shutil.copy2(os.path.join(src,fname), trg)


def create_labels_folder():
    p = 0.65
    src = 'datasets\kitty\orig_labels\\train'
    trg = 'datasets\kitty\orig_labels\\train_65p'

    files = os.listdir(src)
    num_files = len(files)
    files_p = files[:int(num_files * p)]
    # print(files_25p)

    for fname in files_p:
        shutil.copy2(os.path.join(src, fname), trg)


if __name__ == '__main__':
    create_images_folder()
    # create_labels_folder()
