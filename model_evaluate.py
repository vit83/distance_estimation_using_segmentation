from model import attention_unet
from utils import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import collections
import seg_loss_func
from correlation_evaluate import estimate_range_cross_correlation_method


def main():
    # set seed
    batch_size = 1
    seed_val = 100
    seg_2_range_th = 10
    compare_to_correlation = False
    seed_everything(seed_val)
    number_of_targets = 10
    number_of_classes = number_of_targets + 1  # 1 for no target
    dataset_name = "dataset/simple_dataset.npy"
    train_set, val_set, test_set = create_dataset(dataset_name)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    validation_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # get device
    device = get_device()

    seg_model = attention_unet.unet_with_attention(net_in_channels=1, net_out_channels=number_of_classes).to(device)
    seg_model = load_model(seg_model, "segment_model.pth")
    avg_err, avg_false_target_rate, miss_detection_rate, false_target_dict, miss_detect_dict, target_counters = evaluate_model(seg_model, validation_loader, device, seg_2_range_th, compare_to_correlation)
    print("mse {0:.2e} ,false target {1:.2e} miss detected {2:.2e} ".format(avg_err, avg_false_target_rate, miss_detection_rate))
    print("false target statistic")
    for number_of_targets in false_target_dict:
        number_of_false_targets_per_target = len(false_target_dict[number_of_targets])
        number_of_samples_per_target = target_counters[number_of_targets]
        err_percent = number_of_false_targets_per_target / number_of_samples_per_target * 100
        print("for {} targets ".format(number_of_targets))
        print("\t number of false targets {0:d} out of {1:d} {2:.2f} %".format(number_of_false_targets_per_target, number_of_samples_per_target, err_percent))
        print("\t difference from real number of targets")
        counter = collections.Counter(false_target_dict[number_of_targets])
        print('\t', counter)

    print("miss detect statistic")
    for number_of_targets in miss_detect_dict:
        number_of_miss_detect_targets_per_target = len(miss_detect_dict[number_of_targets])
        number_of_samples_per_target = target_counters[number_of_targets]
        err_percent = number_of_miss_detect_targets_per_target / number_of_samples_per_target * 100

        print("for {} targets ".format(number_of_targets))
        print("\t number of miss detect targets {0:d} out of {1:d} {2:.2f} %".format(number_of_miss_detect_targets_per_target, number_of_samples_per_target, err_percent))
        print("\t difference from real number of targets")
        counter = collections.Counter(miss_detect_dict[number_of_targets])
        print('\t', counter)


def evaluate_model(seg_model, data_loader, device, seg_2_range_th=1, useClassicalDetection=False):
    seg_model.eval()
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
    if useClassicalDetection:
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
        seg_mask_gt = torch.nn.functional.pad(seg_mask_gt, (0, padding), "constant", 0)
        data, seg_mask_gt = data.to(device), seg_mask_gt.to(device)
        net_input = data.unsqueeze(1)
        outputs = seg_model(net_input)
        seg_res = outputs.squeeze(1)
        GT = seg_mask_gt.detach().cpu().numpy()
        seg_res_np = seg_res.detach().cpu().numpy()
        net_input_dis = net_input.squeeze(1).detach().cpu().numpy()
        # seg_res_np = fix_seg_mask(seg_res_np)
        seg_label = np.argmax(seg_res_np, axis=1)
        range_estimation, range_index = seg_2_range(seg_label, threshold=seg_2_range_th)
        range_np_gt = ranges_gt.detach().cpu().numpy()
        number_of_targets = np.sum(range_np_gt > 0)
        # remove zeroes
        real_range = range_np_gt[range_np_gt > 0]
        total_real_targets += number_of_targets
        target_counters[number_of_targets] += 1
        # in case we detected all targets
        target_diff = np.size(range_estimation) - number_of_targets
        range_diff = get_range_diff(real_range, range_estimation)
        range_diff = range_diff[range_diff <= max_range - min_range]
        net_input_dis = np.squeeze(net_input_dis, axis=0)
        corr_range = estimate_range_cross_correlation_method(tx_signal=TxSignal, rx_signal=net_input_dis, range_list_gt=real_range)

        target_diff_corr = np.size(corr_range) - number_of_targets
        # if np.abs(target_diff_corr) > np.abs(target_diff):
        #     we_are_better = True
        #     plt.plot(seg_label[0], color='red')
        #     plt.plot(net_input_dis, alpha=0.8)
        #     plt.show()
        #
        #     seg_label_ = np.squeeze(seg_label, axis=0)
        #     gt_mask = np.squeeze(GT, axis=0)
        #     detect_range_cross_correlation_method(tx_signal=TxSignal, rx_signal=net_input_dis, range_list_gt=real_range, mask_seg=seg_label_, gt_seg_mask=gt_mask)
        #
        #     pass
        # elif np.abs(target_diff_corr) < np.abs(target_diff):
        #     we_are_worse = True
        #     pass
        if target_diff == 0:
            range_error = np.sum(np.abs(range_diff))
            range_err_sum += range_error
        elif target_diff > 0:
            false_target += np.size(range_estimation) - np.size(range_diff)
            false_target_dict[number_of_targets].append(target_diff)
        else:
            miss_detect += np.size(real_range) - np.size(range_diff)
            miss_detect_dict[number_of_targets].append(-target_diff)

        # use classical method for detection in case we failed to see if it works
        if target_diff != 0 and useClassicalDetection:
            # net_input_dis = np.squeeze(net_input_dis, axis=0)
            seg_label = np.squeeze(seg_label, axis=0)
            gt_mask = np.squeeze(GT, axis=0)
            detect_range_cross_correlation_method(tx_signal=TxSignal, rx_signal=net_input_dis, range_list_gt=real_range, mask_seg=seg_label, gt_seg_mask=gt_mask)
            with torch.no_grad():
                loss_fn = seg_loss_func.boundary_with_dice_and_cross_entropy_loss()
                loss = loss_fn(seg_res, seg_mask_gt.long())
                print(loss)

    avg_err = range_err_sum / total_real_targets
    false_target_rate = false_target / total_real_targets
    miss_detect_rate = miss_detect / total_real_targets
    return avg_err, false_target_rate, miss_detect_rate, false_target_dict, miss_detect_dict, target_counters


def seg_2_range(seg_results: np.ndarray, threshold=1):
    # project sonar parameters
    Vp = 343.0
    sample_freq_hz = 300000
    tx_signal_len = 900
    range_min_dist = 1e-5
    range_min_dist_in_ind = (range_min_dist * 2 / Vp) * sample_freq_hz

    min_range = 1.0
    max_range = 10.0
    ind_in_range = 0.5 * Vp / sample_freq_hz

    threshold = np.max([threshold, range_min_dist_in_ind])

    seg_results = seg_results[0]
    delta_seg_results = seg_results[1:] - seg_results[:-1]

    up_seg = np.array([])
    down_seg = np.array([])
    for i in range(len(delta_seg_results)):
        dseg = int(delta_seg_results[i])
        if dseg > 0:
            up_seg = np.append(up_seg, i * np.ones(dseg))
        if dseg < 0:
            down_seg = np.append(down_seg, i * np.ones(-dseg))

    segs = get_ind_segs(up_seg, down_seg, tx_signal_len, threshold)

    if np.size(segs) == 0:
        echo_locations = len(seg_results) * 10
    elif np.size(segs) == 2:
        echo_locations = segs[0, 0]

    else:
        echo_locations = np.squeeze(segs[:, 0])

    delays_sec = (echo_locations + 2) / sample_freq_hz
    ranges = delays_sec * 0.5 * Vp

    if np.size(ranges) == 1:
        ranges = np.array([ranges])
        echo_locations = np.array([echo_locations])

    ranges = np.array([np.max([min_range, r]) for r in ranges if r + ind_in_range >= min_range])
    ranges = np.array([np.min([max_range, r]) for r in ranges if r - ind_in_range <= max_range])

    return ranges, echo_locations


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


def get_ind_segs(up_inds, down_inds, tx_signal_len, threshold):
    up_inds = np.copy(up_inds)
    down_inds = np.copy(down_inds)
    max_ind = 18400

    if type(up_inds) == np.float64:
        up_inds = np.array([up_inds])

    if type(down_inds) == np.float64:
        down_inds = np.array([down_inds])

    if down_inds[0] - up_inds[0] < tx_signal_len - threshold:
        ind_del = np.where(down_inds - up_inds[0] < tx_signal_len - threshold)[0]
        ind_keep = [i for i in range(len(down_inds)) if not i in ind_del]
        down_inds = down_inds[ind_keep]

    if down_inds[-1] - up_inds[-1] < tx_signal_len - threshold:
        ind_del = np.where(down_inds[-1] - up_inds < tx_signal_len - threshold)[0]
        ind_keep = [i for i in range(len(up_inds)) if not i in ind_del]
        up_inds = up_inds[ind_keep]

    if len(up_inds) == 1:
        ind_down_ind = np.argmin(np.abs(up_inds[0] - down_inds))
        segs = np.array([[up_inds[0], down_inds[ind_down_ind]]])
        return segs

    # if len(down_inds) < len(up_inds):
    #    down_inds = np.pad(down_inds, (0, len(up_inds) - len(down_inds) + 1), constant_values=(0, max_ind * 3))

    used = np.zeros(len(down_inds))

    count = 0.0
    segs = np.array([10 * max_ind, 50 * max_ind])
    while np.sum(used) < len(up_inds):
        diff_mat, n_diff_mat = get_segs_for_min_err(up_inds[0], down_inds, tx_signal_len, max_ind, threshold)
        for i in range(1, len(up_inds)):
            diff_ar, n_diff = get_segs_for_min_err(up_inds[i], down_inds, tx_signal_len, max_ind, threshold)
            diff_mat = np.vstack((diff_mat, diff_ar))
            n_diff_mat = np.vstack((n_diff_mat, n_diff))

        min_diff_ind = np.vstack((np.min(diff_mat, axis=1), np.argmin(diff_mat, axis=1)))

        if len(np.unique(min_diff_ind[1, :])) == len(min_diff_ind[1, :]) and count == 0:
            segs = np.transpose(np.vstack((up_inds, down_inds[min_diff_ind[1, :].astype(int)])))
            segs = segs[np.sum(segs, axis=1) < 2 * max_ind]
            break

        if np.min(min_diff_ind[0, :]) > threshold:
            if count == 0:
                segs = np.array([10 * max_ind, 50 * max_ind])
            break
        else:
            ind_up_min = int(np.argmin(min_diff_ind[0, :]))
            ind_down_min = int(min_diff_ind[1, ind_up_min])

            used[ind_down_min] = 1

            if count == 0:
                segs = np.array([up_inds[ind_up_min], down_inds[ind_down_min]])
            else:
                segs = np.vstack((segs, np.array([up_inds[ind_up_min], down_inds[ind_down_min]])))

            up_inds[ind_up_min] = -max_ind * (10 + count / 10)
            down_inds[ind_down_min] = max_ind * (10 + count / 10)
            count += 1

    if np.size(segs) == 2:
        segs = np.reshape(segs, (1, 2))

    for i in range(np.size(segs[:, 0])):
        if segs[i, 1] - segs[i, 0] > 1.5 * tx_signal_len:
            n_segs_in_seg = int(np.around((segs[i, 1] - segs[i, 0] + 0.0) / tx_signal_len))
            seg_size = (segs[i, 1] - segs[i, 0]) / n_segs_in_seg
            for k in range(1, n_segs_in_seg):
                up_ind1 = segs[i, 0] + k * seg_size
                if k < n_segs_in_seg - 1:
                    down_ind1 = segs[i, 0] + (k + 1) * seg_size
                else:
                    down_ind1 = segs[i, 1]
                segs = np.vstack((segs, np.array([up_ind1, down_ind1])))
            segs[i, 1] = segs[i, 0] + seg_size

    if np.size(segs) > 2:
        segs = segs[segs[:, 0].argsort()]

    return segs


def get_segs_for_min_err(up_ind, down_ind_ar, signal_len, max_ind, threshold):
    max_targets = 10
    ar_diff = down_ind_ar - up_ind
    ind_non_rel = np.where(np.abs(0.5 * max_ind - ar_diff) > 0.5 * max_ind)[0]
    ar_diff1 = np.vstack((np.mod(ar_diff, signal_len), signal_len - np.mod(ar_diff, signal_len)))

    diff = np.min(ar_diff1, axis=0)
    n_diff = np.round(ar_diff / signal_len).astype(int)

    min_n_diff = np.max([1, np.min(n_diff)]).astype(int)
    max_n_diff = np.min([np.max(n_diff), max_targets]).astype(int)

    ind_wrong_n = np.where(np.abs(n_diff - 0.5 * max_targets - 0.1) > 0.5 * max_targets)[0]
    ind_non_rel = np.append(ind_non_rel, ind_wrong_n)
    ind_non_rel = np.unique(ind_non_rel)
    ind_rel = [i for i in range(len(ar_diff)) if not i in ind_non_rel]

    n_diff_copy = np.copy(n_diff)
    diff_copy = np.copy(diff)

    n_diff = n_diff[ind_rel]
    diff = diff[ind_rel]

    while np.min(np.append(diff[n_diff == min_n_diff], min_n_diff * threshold + 1)) / min_n_diff > threshold and min_n_diff <= max_n_diff:
        min_n_diff += 1

    diff = diff / n_diff

    if min_n_diff <= max_n_diff:
        diff[n_diff != min_n_diff] = diff[n_diff != min_n_diff] + n_diff[n_diff != min_n_diff] * signal_len

    n_diff = n_diff_copy
    diff_copy[ind_rel] = diff
    diff = diff_copy
    if np.size(ind_non_rel) > 0:
        diff[ind_non_rel] = 2 * max_ind * np.ones(np.size(ind_non_rel))

    return diff, n_diff


def fix_seg_mask(seg_res: np.ndarray):
    seg_res_fixed = np.squeeze(seg_res).copy()
    confidence_level_threshold = 0.9
    # find indecision
    vals = np.zeros(seg_res_fixed.shape[1])
    max_val_per_col = np.max(seg_res_fixed, axis=0)
    arg_max_val_per_col = np.argmax(seg_res_fixed, axis=0)
    cols_below_confidence = np.argwhere(max_val_per_col < confidence_level_threshold)
    target_array_size = np.size(max_val_per_col)
    if np.size(cols_below_confidence) > 0:
        for element_index in cols_below_confidence:
            # get closest column from the left that has high confidence

            if element_index == 0 or element_index == target_array_size - 1:
                continue
            left_target_index = element_index - 1
            right_target_index = element_index + 1
            while right_target_index in cols_below_confidence and right_target_index < target_array_size - 1:
                right_target_index += 1

            left_target_confidence = max_val_per_col[left_target_index]
            left_number_of_targets = arg_max_val_per_col[left_target_index]
            right_target_confidence = max_val_per_col[right_target_index]
            right_number_of_targets = arg_max_val_per_col[right_target_index]
            # prefer more targets
            if right_number_of_targets > left_number_of_targets:
                index_to_take = right_target_index
            else:
                index_to_take = left_target_index
            # fix data by copying
            seg_res_fixed[:, element_index] = seg_res_fixed[:, index_to_take]

    seg_res_fixed = np.expand_dims(seg_res_fixed, 0)
    return seg_res_fixed


def detect_range_cross_correlation_method(tx_signal, rx_signal, range_list_gt, mask_seg, gt_seg_mask):
    tx_sig_middle_index = np.size(tx_signal) // 2
    conv_result = np.correlate(rx_signal, tx_signal, 'same')
    number_of_targets = np.sum(range_list_gt > 0)
    detected_targets_index = np.sort(np.argpartition((conv_result), -number_of_targets)[-number_of_targets:])
    Vp = 343
    # translate range to delay
    delays_sec = range_list_gt * 2 / Vp
    sample_freq_hz = 300000
    # translate delay to index
    target_gt_locations_index = (delays_sec[:number_of_targets] * sample_freq_hz).astype(int) - 1 + tx_sig_middle_index

    index_diff = detected_targets_index - target_gt_locations_index

    bPlot = True
    if np.abs(np.sum(index_diff)) > -1:
        print("detection error [m]")
        print(index_diff)
        if bPlot:
            # plt.plot(tx_signal)
            # plt.show()
            plt.plot(rx_signal)
            plt.plot(mask_seg)
            plt.plot(gt_seg_mask, color='red', linestyle='dotted')
            plt.show()
            plt.plot(conv_result)
            plt.show()


if __name__ == '__main__':
    main()
