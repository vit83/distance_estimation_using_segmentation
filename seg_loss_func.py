import torch.nn as nn
import torch.nn.functional


class boundary_with_dice_and_cross_entropy_loss(nn.Module):
    def __init__(self) -> None:
        super(boundary_with_dice_and_cross_entropy_loss, self).__init__()
        self.boundary_loss = BoundaryLoss()
        self.dice_bce_loss = DiceBCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = 0  # self.boundary_loss(pred, target)
        l2 = self.dice_bce_loss(pred, target)

        return l2


class DiceLoss(nn.Module):

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # compute softmax over the classes axis
        input_soft = torch.nn.functional.softmax(input, dim=1)
        input_soft = torch.swapaxes(input_soft, 1, 2)
        # create the labels one hot tensor
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=input.shape[1])

        # compute the actual dice score
        dims = (1, 2)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)


class DiceBCELoss(nn.Module):

    def __init__(self) -> None:
        super(DiceBCELoss, self).__init__()
        self.eps: float = 1e-6
        self.bce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # compute softmax over the classes axis
        input_soft = torch.nn.functional.softmax(input, dim=1)
        input_soft = torch.swapaxes(input_soft, 1, 2)
        # create the labels one hot tensor
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=input.shape[1])

        # compute the actual dice score
        dims = (1, 2)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)

        dice_loss = torch.mean(1. - dice_score)
        bce_loss = self.bce_loss(input, target)
        return dice_loss * 0.2 + bce_loss * 0.8


class IoULoss(nn.Module):

    def __init__(self) -> None:
        super(IoULoss, self).__init__()
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # compute softmax over the classes axis
        input_soft = torch.nn.functional.softmax(input, dim=1)
        input_soft = torch.swapaxes(input_soft, 1, 2)
        # create the labels one hot tensor
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=input.shape[1])

        # compute the actual dice score
        dims = (1, 2)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        union = cardinality - intersection
        IoU = (intersection + self.eps) / (union + self.eps)
        return torch.mean(1. - IoU)


class FocalLoss(nn.Module):

    def __init__(self) -> None:
        super(FocalLoss, self).__init__()
        self.eps: float = 1e-6
        self.alpha: float = 0.8
        self.gamma: float = 2
        self.bce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce_loss(input, target)
        bce_Exp = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - bce_Exp) ** self.gamma * bce_loss
        loss = torch.mean(focal_loss)
        return loss


class TverskyLoss(nn.Module):

    def __init__(self) -> None:
        super(TverskyLoss, self).__init__()
        self.eps: float = 1e-6
        #  this loss function is weighted by the constants 'alpha' and 'beta' that penalise false positives and false negatives respectively to a higher degree in the loss function as their value is increased
        #   larger βs weigh recall higher than precision (by placing more emphasis on false negatives).

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # compute softmax over the classes axis
        alpha: float = 0.2
        beta: float = 0.8

        input_soft = torch.nn.functional.softmax(input, dim=1)
        input_soft = torch.swapaxes(input_soft, 1, 2)
        # create the labels one hot tensor
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=input.shape[1])

        # compute the actual dice score
        dims = (1, 2)

        tp = torch.sum(input_soft * target_one_hot, dims)

        fp = torch.sum(input_soft * (1. - target_one_hot), dims)
        fn = torch.sum((1. - input_soft) * target_one_hot, dims)

        Tversky = tp / (tp + alpha * fp + beta * fn + self.eps)
        return torch.mean(1. - Tversky)


def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    device = label.device
    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 2)

    return one_hot_label


class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super(BoundaryLoss, self).__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        n, number_Of_classes, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        # one_hot_gt = one_hot(gt, number_Of_classes)
        target_one_hot = torch.nn.functional.one_hot(gt, num_classes=number_Of_classes)
        one_hot_gt = torch.swapaxes(target_one_hot, 1, 2).float()
        # boundary map

        gt_b = torch.nn.functional.max_pool1d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        # ground truth boundary map , only transitions between classes appear in this map
        gt_b = gt_b - (1 - one_hot_gt)

        pred_b = torch.nn.functional.max_pool1d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b = pred_b - (1 - pred)

        dims = (1, 2)
        intersection = torch.sum(pred_b * gt_b, dims)
        cardinality = torch.sum(pred_b + gt_b, dims)

        boundary_dice_score = 2. * intersection / (cardinality + 1e-6)
        return torch.mean(1. - boundary_dice_score)

        # extended boundary map
        gt_b_ext = torch.nn.functional.max_pool1d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = torch.nn.functional.max_pool1d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, number_Of_classes, -1)
        pred_b = pred_b.view(n, number_Of_classes, -1)
        gt_b_ext = gt_b_ext.view(n, number_Of_classes, -1)
        pred_b_ext = pred_b_ext.view(n, number_Of_classes, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss
