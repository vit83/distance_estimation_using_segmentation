import numpy as np
import torch
from torch.utils.data import DataLoader
from data_generator import CReflectorsGenerator, CNumpySonarDataSet, CNumpySonarDataSaver
from torchvision import transforms
import random
import matplotlib.pyplot as plt


def check_dataset(dataset):
    b_size = 100
    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=False)
    data_size = dataset.__len__()
    label_array = np.zeros(10)
    # label_array = np.vstack([label_array, label_array])
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
    seed_val = 9560
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    # db_name = "noise_1snr_train_data_10_targets_max_range_1_10_train_with_segmentation.npy"
    db_name = "mixed_dataset.npy"
    tx_waveform_file_name = "TxWaveform.npy"
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
    prob_multinomial = [1 + k ** 2 for k in range(number_of_target)]
    prob_multinomial = prob_multinomial / np.sum(prob_multinomial)
    overlap_prob = 0.9
    signal_time_sec = 3e-3
    Vp = 343
    number_of_ranges_count = np.random.multinomial(number_of_range_lists, prob_multinomial)
    # assume that there is one target , to-do , model without targets
    min_reflectors = 1
    number_of_ranges_list = [[i + 1 for k in range(number_of_ranges_count[i])] for i in range(len(prob_multinomial))]
    number_of_ranges_list = [x for xs in number_of_ranges_list for x in xs]
    samples_per_range_list = 1
    total_samples = number_of_range_lists * samples_per_range_list * 2
    max_overlap_distance = signal_time_sec * Vp / 2
    test_range = np.array([1.7365, 2.0986, 6.2232, 6.9123, 7.2290, 7.4045, 8.2246, 9.5035, 9.5037, 9.9622])
    atten_test = np.ones_like(test_range)
    tx, rx, range_list, seg_mask, _ = data_generator.create_reflectors_waveform(test_range, atten_test, snr)
    # save tx to file
    np.save(tx_waveform_file_name, tx)

    data_generator.clean()
    data_saver = CNumpySonarDataSaver.NumpyCSonarDataSaver(db_name, total_samples, np.size(rx), np.size(range_list))
    prev_range_list = np.zeros_like(range_list)
    duplicated_val = 0
    for i in range(0, number_of_range_lists):
        if i % 100 == 0:
            print("iteration {} out of {}".format(i, number_of_range_lists))
        # sample number of targets
        # num_of_targets = int(np.round(np.random.uniform(min_reflectors, max_reflectors)))
        num_of_targets = number_of_ranges_list[i]
        # sample range based on number of targets
        # range_list = np.random.uniform(min_range_meter, max_range_meter, num_of_targets)
        #        is_in_list = np.any(np.all(range_list == prev_range_list, axis=1))
        #        if is_in_list:
        #            continue
        #        prev_range_list = np.vstack([prev_range_list, range_list])
        # sample range based on number of targets with overlaps
        range_list = np.zeros(num_of_targets)
        # set initial range
        range_list[0] = np.random.uniform(min_range_meter, max_range_meter)
        for k in range(1, num_of_targets):
            overlap = np.random.binomial(size=1, n=1, p=overlap_prob)
            if overlap == 1:
                # select one of the previous targets
                rand_ind = np.random.randint(0, k)
                range_list[k] = np.clip(range_list[rand_ind] + max_overlap_distance * np.random.normal() / 3, min_range_meter, max_range_meter)
            else:
                range_list[k] = np.random.uniform(min_range_meter, max_range_meter)
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
            # plt.figure()
            # plt.plot(rx_sig)
            # plt.show()
            if valid:
                assert (len(range_list_for_train) == max_reflectors)
                assert (np.sum(range_list_for_train) > 0)
                data_saver.add(rx_sig, range_list_for_train, mask_seg)
            else:
                duplicated_val += 1
            if use_classic_detect_range:
                detect_range_cross_correlation_method(tx_sig, rx_sig, range_list_for_train, mask_seg)

    for i in range(0, number_of_range_lists):
        if i % 100 == 0:
            print("iteration {} out of {}".format(i, number_of_range_lists))
        # sample number of targets
        num_of_targets = int(np.round(np.random.uniform(min_reflectors, max_reflectors)))
        # num_of_targets = number_of_ranges_list[i]
        # sample range based on number of targets
        # range_list = np.random.uniform(min_range_meter, max_range_meter, num_of_targets)
        #        is_in_list = np.any(np.all(range_list == prev_range_list, axis=1))
        #        if is_in_list:
        #            continue
        #        prev_range_list = np.vstack([prev_range_list, range_list])
        # sample range based on number of targets with overlaps
        range_list = np.zeros(num_of_targets)
        range_list[0] = np.random.uniform(min_range_meter, max_range_meter)
        for k in range(1, num_of_targets):
            overlap = np.random.binomial(size=1, n=1, p=overlap_prob)
            overlap = 0
            if overlap == 1:
                rand_ind = np.random.randint(0, k)
                range_list[k] = np.clip(range_list[rand_ind] + max_overlap_distance * np.random.normal() / 3, min_range_meter, max_range_meter)
            else:
                range_list[k] = np.random.uniform(min_range_meter, max_range_meter)
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
            # plt.figure()
            # plt.plot(rx_sig)
            # plt.show()
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

    all_uniqe = check_dataset(dataset)
    if all_uniqe:
        print("data set valid")
    else:
        print("data set invalid")


#    for waveform, labels in train_dataloader:
#        # train the network
#        pass


if __name__ == '__main__':
    main()
    print("done")
