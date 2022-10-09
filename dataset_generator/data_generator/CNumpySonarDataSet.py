import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from numpy.lib.format import open_memmap


class CNumpySonarDataSet(Dataset):

    def __init__(self, data_base_file_name, transform=None, target_transform=None):
        self.numpy_mem_file = open_memmap(data_base_file_name, mode='r', dtype=np.float, )
        self.transform = transform
        self.target_transform = target_transform
        self.data_sample_len = 18391
        self.seg_mask_len = self.data_sample_len
        self.label_len = 10
        self.number_of_samples = int(self.numpy_mem_file.size / (self.label_len + self.data_sample_len + self.seg_mask_len))
        pass
        max_range = 0
        min_range = 11
        # get max and min range
        for index in range(0, self.number_of_samples):
            data_with_label = self.numpy_mem_file[index]
            data = data_with_label[0:self.data_sample_len]
            num_target = np.sum(data_with_label[self.data_sample_len:self.data_sample_len + self.label_len] > 0)
            range_list = data_with_label[self.data_sample_len:self.data_sample_len + num_target]
            if np.max(range_list) > max_range:
                max_range = np.max(range_list)
                self.max_waveform_index = index

            if np.min(range_list) < min_range:
                min_range = np.min(range_list)
                self.min_waveform_index = index

    def __getitem__(self, index):

        data_with_label = self.numpy_mem_file[index]
        data = data_with_label[0:self.data_sample_len]
        label = data_with_label[self.data_sample_len:self.data_sample_len + self.label_len]
        seg_mask = data_with_label[self.data_sample_len + self.label_len:]
        if self.transform is not None:
            data = self.transform(data)
        data = torch.from_numpy(data).float()
        seg_mask = torch.from_numpy(seg_mask).float()
        label = torch.from_numpy(label)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label, seg_mask

    def __len__(self):
        return self.number_of_samples

    def get_max_range_sample(self):
        return self.max_waveform_index

    def get_min_range_sample(self):
        return self.min_waveform_index
