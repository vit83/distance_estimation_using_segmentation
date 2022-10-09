
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from PIL import Image
from scipy.signal import spectrogram, stft
from data_generator import CNumpySonarDataSet


def main():
    db_name = "..//dataset//simple_dataset.npy"
    dataset = CNumpySonarDataSet.CNumpySonarDataSet(db_name, transform=None)

    data_size = dataset.__len__()
    print("db sample size", data_size)

    train_set, val_set = torch.utils.data.random_split(dataset, [int(data_size * 0.9), int(data_size * 0.1)])
    print("train db sample size ", train_set.__len__())
    print("test db sample size ", val_set.__len__())
    train_dataloader = DataLoader(train_set, batch_size=1, shuffle=False)
    for waveform, labels, seg_mask in train_dataloader:
        # f, t, Sxx = spectrogram(x=waveform, fs=300000, nperseg=128, noverlap=120, nfft=128)
        print("range ", labels)
        plt.figure()
        plt.plot(waveform[0])
        plt.plot(seg_mask[0])
        plt.show()


if __name__ == '__main__':
    main()
    print("done")
