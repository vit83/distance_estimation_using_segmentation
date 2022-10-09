from utils import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import collections


def main():
    # set seed
    batch_size = 1
    seed_val = 100
    seed_everything(seed_val)
    dataset_name = "dataset/simple_dataset.npy"
    train_set, val_set, test_set = create_dataset(dataset_name)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    validation_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # get device
    device = get_device()

    avg_err, avg_false_target_rate, miss_detection_rate, false_target_dict, miss_detect_dict, target_counters = detect_using_correlation(validation_loader, device)
    print("mse {} ,false target {} miss detected {} ".format(avg_err, avg_false_target_rate, miss_detection_rate))
    print("false target statistic")
    for number_of_targets in false_target_dict:
        number_of_false_targets_per_target = len(false_target_dict[number_of_targets])
        number_of_samples_per_target = target_counters[number_of_targets]
        err_percent = number_of_false_targets_per_target / number_of_samples_per_target * 100
        print("for {} targets ".format(number_of_targets))
        print("\t number of false targets {} out of {} {} %".format(number_of_false_targets_per_target, number_of_samples_per_target, err_percent))
        print("\t difference from real number of targets")
        counter = collections.Counter(false_target_dict[number_of_targets])
        print('\t', counter)

    print("miss detect statistic")
    for number_of_targets in miss_detect_dict:
        number_of_miss_detect_targets_per_target = len(miss_detect_dict[number_of_targets])
        number_of_samples_per_target = target_counters[number_of_targets]
        err_percent = number_of_miss_detect_targets_per_target / number_of_samples_per_target * 100

        print("for {} targets ".format(number_of_targets))
        print("\t number of miss detect targets {} out of {} {} %".format(number_of_miss_detect_targets_per_target, number_of_samples_per_target, err_percent))
        print("\t difference from real number of targets")
        counter = collections.Counter(miss_detect_dict[number_of_targets])
        print('\t', counter)


def detect_using_correlation(data_loader, device):
    total_real_targets = 0
    range_err_sum = 0
    false_target = 0
    miss_detect = 0
    max_range = 10.0
    min_range = 1.0

    false_target_dict = {}
    miss_detect_dict = {}
    target_counters = {}
    for i in range(1, 11):
        false_target_dict[i] = []
        miss_detect_dict[i] = []
        target_counters[i] = 0

    TxSignal = np.load('dataset/TxWaveform.npy')
    sample_num = 0
    for data, ranges_gt, seg_mask_gt in data_loader:
        data, seg_mask_gt = data.to(device), seg_mask_gt.to(device)
        data_size = data.size()[1]
        sample_num += 1
        # net input need to be power of 2
        # power_of_two = 2 ** (data_size - 1).bit_length() // 2

        # net input should be multiple of 16 pad if required
        padding = 0
        if data_size % 16 != 0:
            padding = (data_size // 16 + 1) * 16 - data_size
        data = torch.nn.functional.pad(data, (0, padding), "constant", 0)
        range_np_gt = ranges_gt.detach().cpu().numpy()
        number_of_targets = np.sum(range_np_gt > 0)
        # remove zeroes
        real_range = range_np_gt[range_np_gt > 0]
        total_real_targets += number_of_targets
        target_counters[number_of_targets] += 1
        net_input_np = data.squeeze(0).detach().cpu().numpy()

        range_estimation = estimate_range_cross_correlation_method(tx_signal=TxSignal, rx_signal=net_input_np, range_list_gt=real_range)

        target_diff = np.size(range_estimation) - number_of_targets

        # range_diff = get_range_diff(real_range, range_estimation)
        # range_diff = range_diff[range_diff <= max_range - min_range]
        # target_diff = np.size(range_estimation) - np.size(range_diff)
        if target_diff == 0:
            range_diff = range_estimation - real_range
            range_error = np.sum(np.abs(range_diff))
            range_err_sum += range_error
        elif target_diff > 0:
            false_target += target_diff
            false_target_dict[number_of_targets].append(target_diff)
        else:
            miss_detect += -target_diff
            miss_detect_dict[number_of_targets].append(-target_diff)

    avg_err = range_err_sum / total_real_targets
    false_target_rate = false_target / total_real_targets
    miss_detect_rate = miss_detect / total_real_targets
    return avg_err, false_target_rate, miss_detect_rate, false_target_dict, miss_detect_dict, target_counters


def get_range_diff(gt_ranges, est_ranges):
    gt_ranges = np.copy(gt_ranges)
    est_ranges = np.copy(est_ranges)
    gt_ranges = gt_ranges[gt_ranges != 0]
    est_ranges = est_ranges[est_ranges != 0]
    max_range = 10.0

    if type(gt_ranges) == np.float64:
        gt_ranges = np.array([gt_ranges])

    if type(est_ranges) == np.float64:
        est_ranges = np.array([est_ranges])

    if len(gt_ranges) == 1:

        if np.size(est_ranges) == 0:
            range_diff = np.array([])
        elif np.size(est_ranges) == 1:
            range_diff = np.abs(gt_ranges[0] - est_ranges[0])
        else:
            range_diff = np.min(np.abs(gt_ranges[0] - est_ranges))
        return range_diff

    if len(est_ranges) < len(gt_ranges):
        est_ranges = np.pad(est_ranges, (0, len(gt_ranges) - len(est_ranges) + 1), constant_values=(0, max_range * 3))

    used = np.zeros(len(est_ranges))
    range_diff = -np.ones(len(gt_ranges))

    count = 0.0
    while np.sum(used) < len(gt_ranges):
        diff_mat = np.abs(gt_ranges[0] - est_ranges)
        for i in range(1, len(gt_ranges)):
            diff_mat = np.vstack((diff_mat, np.abs(gt_ranges[i] - est_ranges)))

        min_diff_ind = np.vstack((np.min(diff_mat, axis=1), np.argmin(diff_mat, axis=1)))

        if len(np.unique(min_diff_ind[1, :])) == len(min_diff_ind[1, :]) and count == 0:
            range_diff = np.squeeze(min_diff_ind[0, :])
            return range_diff

        ind_gt_min = int(np.argmin(min_diff_ind[0, :]))
        ind_est_min = int(min_diff_ind[1, ind_gt_min])

        used[ind_est_min] = 1
        range_diff[ind_gt_min] = np.min(min_diff_ind[0, :])

        gt_ranges[ind_gt_min] = -max_range * (10 + count / 10)
        est_ranges[ind_est_min] = max_range * (10 + count / 10)
        count += 1

    return range_diff


def estimate_range_cross_correlation_method(tx_signal, rx_signal, range_list_gt, display_detection_error=False):
    tx_sig_middle_index = np.size(tx_signal) // 2
    number_of_gt_targets = np.sum(range_list_gt > 0)

    normalize = False
    if normalize:
        norm_rx_signal = np.linalg.norm(rx_signal)
        rx_signal = rx_signal / norm_rx_signal
        norm_tx_signal = np.linalg.norm(tx_signal)
        tx_signal = tx_signal / norm_tx_signal

    conv_result = np.correlate(rx_signal, tx_signal, 'same')

    # half of tx signal energy
    detection_th = 0.5 * np.dot(tx_signal, tx_signal)
    number_of_detected_targets = np.sum(conv_result > detection_th)
    if number_of_detected_targets != number_of_gt_targets and display_detection_error:
        print("real targets {}".format(number_of_gt_targets))
        print("detected targets {}".format(number_of_detected_targets))
        plt.plot(conv_result)
        plt.show()

    # get k argmax values
    detected_targets_index = np.sort(np.argpartition((conv_result), -number_of_detected_targets)[-number_of_detected_targets:])
    Vp = 343
    sample_freq_hz = 300000
    detected_targets_delay = (detected_targets_index - tx_sig_middle_index + 1) * (1 / sample_freq_hz)
    estimated_range = detected_targets_delay * Vp * 0.5

    return estimated_range


if __name__ == '__main__':
    main()
