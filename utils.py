import torch
import torch.nn.init as init
import CNumpySonarDataSet
import os
import random
import numpy as np


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Cuda capability")
        print(torch.cuda.get_device_capability())
        print(torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
    return device


def print_model_params(model):
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of training parameters", pytorch_total_params)


def predict(model, data):
    with torch.no_grad():
        model.eval()
        outputs = model(data)
        return outputs


def create_dataset(dataset_name):
    dataset = CNumpySonarDataSet.CNumpySonarDataSet(dataset_name, transform=None, target_transform=None)
    data_size = dataset.__len__()
    print("db sample size", data_size)
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 1 - train_ratio - val_ratio

    train_size = int(data_size * train_ratio)
    validation_size = int(data_size * val_ratio)
    test_size = data_size - train_size - validation_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])
    # train_set = dataset
    print("train db sample size ", train_set.__len__())
    print("validation db sample size ", val_set.__len__())
    print("test db sample size ", test_set.__len__())
    return train_set, val_set, test_set


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
