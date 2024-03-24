from config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, NUM_WORKERS, ROOT,
)
from model import create_model
from custom_utils import Averager, SaveBestModel, save_model, save_loss_plot
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)

import torch
import matplotlib.pyplot as plt
import time
from torchmetrics.detection import MeanAveragePrecision



plt.style.use('ggplot')

# function for running training iterations
def train(train_data_loader, model):

    print('Training')
    global train_itr
    global train_loss_list
    model.train()
    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)

        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()

        train_itr += 1
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list

# function for running validation iterations
def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list
    global metric
    global map_dict_list
    model.eval()

    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            # loss_dict = model(images, targets)


            preds = model(images)

        # losses = sum(loss for loss in loss_dict.values())
        # loss_value = losses.item()
        # val_loss_list.append(loss_value)
        #
        # val_loss_hist.send(loss_value)

        metric.update(preds, targets)


        val_itr += 1

        # update the loss value beside the progress bar for each iteration
        # prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    map_dict_list.append(metric.compute())

    return val_loss_list

if __name__ == '__main__':
    train_dataset = create_train_dataset()
    valid_dataset = create_valid_dataset()
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # initialize the model and move to the computation device
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)

    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    metric = MeanAveragePrecision(class_metrics=True)
    metric.to(DEVICE)
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []
    map_dict_list = []

    # name to save the trained model with
    MODEL_NAME = 'model'

    # whether to show transformed images from data loader or not
    if VISUALIZE_TRANSFORMED_IMAGES:
        from custom_utils import show_tranformed_image
        show_tranformed_image(train_loader)

    # initialize SaveBestModel class
    save_best_model = SaveBestModel()

    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()

        # start timer and carry out training and validation
        start = time.time()
        # train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)
        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch+1} validation loss: {val_loss_hist.value:.3f}")   
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        # save the best model till now if we have the least loss in the...
        # ... current epoch
        save_best_model(
            val_loss_hist.value, epoch, model, optimizer
        )
        # save the current epoch model
        save_model(epoch, model, optimizer)

        # save loss plot
        # save_loss_plot(OUT_DIR, train_loss, val_loss)
        
        # sleep for 5 seconds after each epoch
        time.sleep(5)

    all_epoch_map_stats = dict()
    for i, elem in enumerate(map_dict_list):
        all_epoch_map_stats[i] = elem

    STATS_DIR = ROOT + 'image-detection-on-kitty-dataset\Faster_RCNN\\runs\\run3\\'
    try:
        stats_file = open(STATS_DIR + 'map_stats.txt', 'wt')
        stats_file.write(str(all_epoch_map_stats))
        stats_file.close()

    except:
        print("Unable to write to file")

    fig_, ax_ = metric.plot()
    PLOT_DIR = ROOT + 'image-detection-on-kitty-dataset\Faster_RCNN\\runs\\run3\\'
    fig_.savefig(PLOT_DIR + 'progress.png')
