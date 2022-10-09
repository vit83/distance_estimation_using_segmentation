import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional
from torch.utils.data import DataLoader
import time

import model_metrcis
from model import attention_unet
from utils import *
import seg_loss_func


def main():
    # set seed
    seed_val = 100
    seed_everything(seed_val)
    # get device
    device = get_device()
    number_of_targets = 10
    number_of_classes = number_of_targets + 1  # 1 for no target
    dataset_name = "dataset/simple_dataset.npy"
    seg_model = attention_unet.unet_with_attention(net_in_channels=1, net_out_channels=number_of_classes).to(device)

    # print model parameters
    print("model params")
    print_model_params(seg_model)

    # create data set
    train_set, val_set, test_set = create_dataset(dataset_name)

    # set model loss and optimizer

    learning_rate = 1e-3
    fWeight_decay = 0
    betas = (0.7, 0.9)
    lr_scheduler_step_size = 1
    lr_scheduler_gamma = 0.85

    bce_and_dice_loss = seg_loss_func.DiceBCELoss()
    # assign loss function
    loss_func = bce_and_dice_loss
    optimizer = torch.optim.Adam(seg_model.parameters(), lr=learning_rate, betas=betas, weight_decay=fWeight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)

    # train model
    num_epochs = 60
    batch_size = 34

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    validation_loader = DataLoader(val_set, batch_size=18, shuffle=True)

    loss_ar = np.array([])
    loss_val_ar = np.array([])
    dice_score_ar = np.array([])

    for epoch in range(num_epochs):
        start_time = time.time()
        loss = train_one_epoch(seg_model, optimizer, train_loader, device, loss_func)
        # Validate the model
        loss_val = val_one_epoch(seg_model, validation_loader, device, loss_func)
        dice_score_validation = calc_dice_score(seg_model, validation_loader, number_of_classes, device)
        loss_ar = np.append(loss_ar, loss)
        loss_val_ar = np.append(loss_val_ar, loss_val)
        dice_score_ar = np.append(dice_score_ar, dice_score_validation)
        scheduler.step()
        end_time = time.time() - start_time
        print("Epoch: {}, Loss: {}, loss val: {}, Time[sec]: {}".format(epoch, round(loss, 4), round(loss_val, 4),
                                                                        round(end_time, 2)))
        print("validation dice score {}".format(dice_score_validation))

    save_model(seg_model, "segment_model.pth")

    print("Loss array")
    print(loss_ar)
    print("Loss val array")
    print(loss_val_ar)
    print("Dice Score array")
    print(dice_score_ar)

    plt.rcParams["figure.figsize"] = (20, 15)

    plt.figure()
    plt.plot(loss_ar, label='Training Loss')
    plt.plot(loss_val_ar, color='red', linestyle='dotted', label='Validation Loss')
    plt.xlabel("Num Epoch", fontsize=26)
    plt.xticks(fontsize=20)
    plt.ylabel("Loss Value", fontsize=26)
    plt.yticks(fontsize=20)
    plt.yscale("log")
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=26)
    plt.savefig('training_loss.png')
    plt.show()

    plt.figure()
    plt.plot(dice_score_ar)
    plt.xlabel("Num Epoch", fontsize=26)
    plt.xticks(fontsize=20)
    plt.ylabel("Validation Dice Score", fontsize=26)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.savefig('dice_score.png')
    plt.show()


def train_one_epoch(seg_model, optimizer, train_loader, device, loss_func):
    seg_model.train()
    count = 0
    loss_sum = 0
    for data, labels, seg_mask in train_loader:
        data_size = data.size()[1]
        # net input should be multiple of 16 pad if required
        padding = 0
        if data_size % 16 != 0:
            padding = (data_size // 16 + 1) * 16 - data_size
        data = torch.nn.functional.pad(data, (0, padding), "constant", 0)
        seg_mask = torch.nn.functional.pad(seg_mask, (0, padding), "constant", 0)

        data, seg_mask = data.to(device), seg_mask.to(device)
        net_input = data.unsqueeze(1)
        outputs = seg_model(net_input)
        seg_mask = seg_mask.long()
        loss = loss_func(outputs, seg_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        count += 1
    return loss_sum / count


def val_one_epoch(seg_model, val_loader, device, loss_func):
    seg_model.eval()
    count = 0
    loss_sum = 0
    with torch.no_grad():
        for data, labels, seg_mask in val_loader:
            data_size = data.size()[1]
            # net input should be multiple of 16 pad if required
            padding = 0
            if data_size % 16 != 0:
                padding = (data_size // 16 + 1) * 16 - data_size
            data = torch.nn.functional.pad(data, (0, padding), "constant", 0)
            seg_mask = torch.nn.functional.pad(seg_mask, (0, padding), "constant", 0)

            data, seg_mask = data.to(device), seg_mask.to(device)
            net_input = data.unsqueeze(1)
            outputs = seg_model(net_input)
            seg_mask = seg_mask.long()
            loss = loss_func(outputs, seg_mask)

            loss_sum += loss.item()
            count += 1
    return loss_sum / count


def calc_dice_score(seg_model, data_loader, number_of_classes, device):
    length = 0
    dice_coef = 0
    seg_model.eval()
    with torch.no_grad():
        for data, labels, seg_mask in data_loader:
            data, seg_mask = data.to(device), seg_mask.to(device)
            data_size = data.size()[1]
            # net input should be multiple of 16 pad if required
            padding = 0
            if data_size % 16 != 0:
                padding = (data_size // 16 + 1) * 16 - data_size
            data = torch.nn.functional.pad(data, (0, padding), "constant", 0)
            seg_mask = torch.nn.functional.pad(seg_mask, (0, padding), "constant", 0)
            data, seg_mask = data.to(device), seg_mask.to(device)
            seg_mask = seg_mask.long()
            net_input = data.unsqueeze(1)
            outputs = seg_model(net_input)
            seg_res = outputs.squeeze(1)
            segmentation_result = torch.argmax(seg_res, dim=1)

            ground_truth = seg_mask
            segmentation_result = segmentation_result.detach().cpu().numpy()
            ground_truth = ground_truth.detach().cpu().numpy()
            length += 1
            dice_coef += model_metrcis.dice_coef_multilabel(ground_truth.flatten(), segmentation_result.flatten(), number_of_classes)

            pass

    avg_dice = dice_coef / length
    return avg_dice


if __name__ == '__main__':
    main()
