
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

def compute_acc(labels, x_labels, nb_batch):
    """ compute the classification accuracy
    Args:
        labels: ground truth label
        x_labels: predicted label
        nb_batch: batch size
    """
    N = int(nb_batch * 10)
    pre_labels = np.zeros(N)
    real_labels = np.zeros(N)
    c = 0
    for i in range(nb_batch):
        for j in range(x_labels.shape[1]): # x_labels.shape: [bs, 10, 29]
            pre_labels[c] = np.argmax(x_labels[i, j, :]) #
            real_labels[c] = np.argmax(labels[i, j, :])
            c += 1
    target_names = []
    for i in range(29):
        target_names.append("class" + str(i))

    return accuracy_score(real_labels, pre_labels)


def compute_accuracy_supervised(is_event_scores, event_scores, labels, threshold=0.5):
    # labels = labels[:, :, :-1]  # 28 denote background
    targets = labels.argmax(-1)
    # pos pred
    scores_pos_ind = is_event_scores > threshold
    scores_mask = scores_pos_ind == 0
    event_class = event_scores.argmax(-1)    # foreground classification
    pred = scores_pos_ind.astype(np.int64)
    pred *= event_class[:, None]
    # add mask
    pred[scores_mask] = 28 # 28 denotes bg
    # correct = np.equal(pred, targets)
    # correct_num = correct.sum().astype(np.float64)
    # acc = correct_num * (1. / correct.numel())
    targets = targets.reshape(-1)
    pred = pred.reshape(-1)

    acc = accuracy_score(targets, pred)

    return acc


def compute_confusion_matrix(is_event_scores, event_scores, labels):
    # labels = labels[:, :, :-1]  # 28 denote background
    targets = 1 - labels[:, :, -1]
    scores_pos_ind = is_event_scores > 0.5

    t = []
    s = []
    for target in targets:
        t += target.tolist()
    for score in scores_pos_ind:
        s += score.tolist()

    print(confusion_matrix(t, s, normalize='true'))

    tn, fp, fn, tp = confusion_matrix(t, s, normalize='true').ravel()
    print('tn: ', tn)
    print('fp: ', fp)
    print('fn: ', fn)
    print('tp: ', tp)


def AVPSLoss(av_simm, soft_label):
    """audio-visual pair similarity loss for fully supervised setting,
    please refer to Eq.(8, 9) in our paper.
    """
    # av_simm: [bs, 10]
    relu_av_simm = F.relu(av_simm)
    sum_av_simm = torch.sum(relu_av_simm, dim=-1, keepdim=True)
    avg_av_simm = relu_av_simm / (sum_av_simm + 1e-8)
    loss = nn.MSELoss()(avg_av_simm, soft_label)
    return loss
