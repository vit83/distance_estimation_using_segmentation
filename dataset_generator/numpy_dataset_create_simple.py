import numpy as np
import torch
from torch.utils.data import DataLoader
from data_generator import CReflectorsGenerator, CNumpySonarDataSet, CNumpySonarDataSaver
import random
import matplotlib.pyplot as plt


def check_dataset(dataset):
    b_size = 100
    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=False)
    label_array = np.zeros(10)
    i = 0
    for data, label, seg_mask in dataloader:
        label_array = np.vstack([label_array, label])
        i = i + b_size
    unique_labels = np.unique(label_array, axis=0)
    all_unique = np.shape(unique_labels)[0] == i + 1
    print("number of non unique", i + 1 - np.shape(unique_labels)[0])
    return all_unique


def detect_range_cross_correlation_method(tx_signal, rx_signal, range_list, mask_seg_estimation_gt):
    tx_sig_middle_index = np.size(tx_signal) // 2
    conv_result = np.correlate(rx_signal, tx_signal, 'same')
    number_of_targets = np.sum(range_list > 0)
    detected_targets_index = np.sort(np.argpartition((conv_result), -number_of_targets)[-number_of_targets:])
    Vp = 343
    delays_sec = range_list * 2 / Vp
    sample_freq_hz = 300000
    echo_locations = (delays_sec[:number_of_targets] * sample_freq_hz).astype(int) - 1 + tx_sig_middle_index

    err = detected_targets_index - echo_locations

    bPlot = True
    if np.abs(np.sum(err)) > 2:
        print("detection error [m]")
        print(err)
        if bPlot:
            # plt.plot(tx_signal)
            # plt.show()
            plt.plot(rx_signal)

            plt.plot(mask_seg_estimation_gt)
            plt.show()
            plt.plot(conv_result)
            plt.show()


def main():
    # db should be the same between runs
    seed_val = 123
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    db_name = "..//dataset//simple_dataset.npy"
    tx_waveform_file_name = "..//dataset//TxWaveform.npy"
    # const parameters , do not change
    max_reflectors = 10
    data_generator = CReflectorsGenerator.CReflectorsGenerator(max_reflectors)
    # config parameters
    number_of_target = 10  # change this number only
    snr = 10  # 10.0 # snr is ratio not db!!!!!
    number_of_range_lists = 10000
    min_range_meter = 1
    max_range_meter = 10
    use_classic_detect_range = False
    signal_time_sec = 3e-3
    Vp = 343
    # assume that there is one target , to-do , model without targets
    min_reflectors = 1
    samples_per_range_list = 1
    total_samples = number_of_range_lists * samples_per_range_list
    max_overlap_distance = signal_time_sec * Vp / 2
    test_range = np.array([1.7365, 2.0986, 6.2232, 6.9123, 7.2290, 7.4045, 8.2246, 9.5035, 9.5037, 9.9622])
    atten_test = np.ones_like(test_range)
    tx, rx, range_list, seg_mask, _ = data_generator.create_reflectors_waveform(test_range, atten_test, snr)
    # save tx to file
    np.save(tx_waveform_file_name, tx)

    data_generator.clean()
    data_saver = CNumpySonarDataSaver.NumpyCSonarDataSaver(db_name, total_samples, np.size(rx), np.size(range_list))
    duplicated_val = 0
    for i in range(0, number_of_range_lists):
        if i % 100 == 0:
            print("iteration {} out of {}".format(i, number_of_range_lists))
        # sample number of targets
        num_of_targets = int(np.round(np.random.uniform(min_reflectors, max_reflectors)))
        # set initial range
        range_list = np.random.uniform(min_range_meter, max_range_meter, size=num_of_targets)
        # assume no attenuation
        attenuation_list = np.ones_like(range_list) * 1  # np.random.normal()
        # generate and add samples
        for j in range(0, samples_per_range_list):
            tx_sig, rx_sig, range_list_for_train, mask_seg, valid = data_generator.create_reflectors_waveform(range_list, attenuation_list, snr)
            while not valid:
                range_list = np.random.uniform(min_range_meter, max_range_meter, num_of_targets)
                tx_sig, rx_sig, range_list_for_train, mask_seg, valid = data_generator.create_reflectors_waveform(range_list, attenuation_list, snr)
                if not valid:
                    print("data invalid retrying")
                    pass
            if valid:
                assert (len(range_list_for_train) == max_reflectors)
                assert (np.sum(range_list_for_train) > 0)
                data_saver.add(rx_sig, range_list_for_train, mask_seg)
            else:
                duplicated_val += 1
            if use_classic_detect_range:
                detect_range_cross_correlation_method(tx_sig, rx_sig, range_list_for_train, mask_seg)

    # check the created dataset
    dataset = CNumpySonarDataSet.CNumpySonarDataSet(db_name, transform=None)

    data_size = dataset.__len__()
    print("db sample size", data_size)
    assert total_samples == data_size
    train_set, val_set = torch.utils.data.random_split(dataset, [int(data_size * 0.9), int(data_size * 0.1)])
    print("train db sample size ", train_set.__len__())
    print("test db sample size ", val_set.__len__())
    data, label, seg_mask = dataset[99]
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(val_set, batch_size=64, shuffle=True)
    #    for data in train_dataloader:
    #        print(data)

    train_features, train_labels, seg_mask = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    all_unique = check_dataset(dataset)
    if all_unique:
        print("data set valid")
    else:
        print("data set invalid")


if __name__ == '__main__':
    main()
    print("done")
