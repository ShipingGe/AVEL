from __future__ import print_function
import os
import time
import torch
import torch.nn as nn
import numpy as np
import random

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, CosineAnnealingWarmRestarts

from dataloader import get_loader
from fully_model import MAML
from measure import compute_acc, AVPSLoss, compute_accuracy_supervised

import warnings

warnings.filterwarnings("ignore")
import argparse

parser = argparse.ArgumentParser(description='Fully supervised AVE localization')

# data
parser.add_argument('--model_name', type=str, default='PSP', help='model name')

parser.add_argument('--nb_epoch', type=int, default=300, help='number of epoch')
parser.add_argument('--train_batch_size', type=int, default=128, help='number of training batch size')
parser.add_argument('--eval_batch_size', type=int, default=128, help='number of eval batch size')
parser.add_argument('--save_epoch', type=int, default=1, help='number of epoch for saving models')
parser.add_argument('--check_epoch', type=int, default=1,
                    help='number of epoch for checking accuracy of current models during training')
parser.add_argument('--threshold', type=float, default=0.5, help='threshold for event relevance')

parser.add_argument('--trained_model_path', type=str, default=None, help='path to the trained model')
parser.add_argument('--train', action='store_true', default=True, help='train a new model')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--seed', type=int, default=2, help="random seed for initialization")
parser.add_argument('--num_layers', type=int, default=1, help="num of the transformer blocks")

parser.add_argument('--lambda_1', type=float, default=1, help="weight for loss_1")
parser.add_argument('--lambda_2', type=float, default=1, help="weight for loss_2")
parser.add_argument('--lambda_3', type=float, default=0.01, help="weight for loss_3")
parser.add_argument('--lambda_4', type=float, default=0.1, help="weight for loss_4")
parser.add_argument('--lambda_5', type=float, default=0.1, help="weight for loss_5")

parser.add_argument('--n_gpus', type=int, default=2, help="num of gpus for training")
parser.add_argument('--bank_size', type=int, default=32, help="size of the memory bank.")

parser.add_argument('--margin', type=float, default=0.7, help="margin for the contrastive learning.")


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpus > 0:
        torch.cuda.manual_seed_all(args.seed)


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def train(args, net_model):

    epoch_l = []
    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0

    # prepare dataloader
    train_loader, val_loader, test_loader = get_loader(args)

    optimizer = optim.Adam(net_model.parameters(), lr=args.lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.nb_epoch)

    loss_fn_ce = nn.CrossEntropyLoss().cuda()
    loss_fn_bce = nn.BCELoss().cuda()
    loss_fn_uni_ce = nn.CrossEntropyLoss(reduction='none').cuda()

    net_model.zero_grad()

    rand_idx = str(int(time.time()))
    print('training idx: ', rand_idx)

    for epoch in range(args.nb_epoch):
        net_model.train()

        epoch_loss = 0
        epoch_loss_cls = 0
        epoch_loss_loc = 0
        epoch_loss_uni = 0
        epoch_loss_pt = 0
        epoch_loss_con = 0
        n = 0

        for i, data in enumerate(train_loader):
            audio_inputs, video_inputs, onehot_labels = data

            audio_inputs = audio_inputs.cuda()
            video_inputs = video_inputs.cuda()
            onehot_labels = onehot_labels.cuda()

            bg_preds, evn_preds, \
            audio_evn_preds, video_evn_preds, \
            rf_bg_pred, rf_evn_pred = net_model(audio_inputs, video_inputs, onehot_labels)  # shape: out_prob: [bs, 10, 29], score_max: [bs, 29]

            # generating bg labels and event labels
            labels_foreground = onehot_labels[:, :, :-1]  # shape: [bs, 10, 28]
            labels_bg, labels_evn = labels_foreground.max(-1)
            labels_event, _ = labels_evn.max(-1)

            loss_bg = args.lambda_1 * loss_fn_bce(bg_preds, labels_bg)
            loss_evn = args.lambda_2 * loss_fn_ce(evn_preds, labels_event)

            B, T, num_classes = audio_evn_preds.shape
            uni_modal_cls_loss = loss_fn_uni_ce(audio_evn_preds.reshape(B*T, -1), labels_evn.reshape(-1)) + \
                       loss_fn_uni_ce(video_evn_preds.reshape(B*T, -1), labels_evn.reshape(-1))
            loss_u = args.lambda_3 * (uni_modal_cls_loss * labels_bg.reshape(-1)).mean()

            prototype_loss = loss_fn_ce(rf_evn_pred, labels_event)
            loss_p = args.lambda_4 * prototype_loss.mean()

            contrastive_loss = rf_bg_pred.mean()
            loss_c = args.lambda_5 * contrastive_loss.mean()

            loss = loss_bg + loss_evn + loss_u + loss_p + loss_c

            epoch_loss += loss.cpu().data.numpy()
            epoch_loss_loc += loss_bg.cpu().data.numpy()
            epoch_loss_cls += loss_evn.cpu().data.numpy()
            epoch_loss_uni += loss_u.cpu().data.numpy()
            epoch_loss_pt += loss_p.cpu().data.numpy()
            epoch_loss_con += loss_c.cpu().data.numpy()

            optimizer.zero_grad()
            loss.backward()

            # clip_grad_norm_(net_model.parameters(), 10.0)
            optimizer.step()

            n = n + 1
        scheduler.step()

        epoch_l.append(epoch_loss)
        onehot_labels = onehot_labels.cpu().data.numpy()
        bg_preds = bg_preds.cpu().data.numpy()
        evn_preds = evn_preds.cpu().data.numpy()
        acc = compute_accuracy_supervised(bg_preds, evn_preds, onehot_labels, args.threshold)

        print(
            "Epoch {%s}  lr: {%.6f} | Total_loss: [{%.4f}] loss_cls: [{%.4f}] | loss_loc: [{%.4f}] "
            "| loss_uni: [{%.4f}] | loss_pt: [{%.4f}] | loss_con: [{%.4f}] | training_acc {%.4f}" \
            % (str(epoch), optimizer.param_groups[0]['lr'], (epoch_loss) / n, epoch_loss_cls / n,
               epoch_loss_loc / n, epoch_loss_uni / n, epoch_loss_pt / n, epoch_loss_con / n, acc))

        if epoch % args.save_epoch == 0 and epoch != 0:
            val_acc = val(args, net_model, val_loader)
            # writer.add_scalar('Eval/Acc', val_acc, epoch)
            print('val accuracy:', val_acc, 'epoch=', epoch)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                print('best val accuracy:', best_val_acc)
                print('best val accuracy: {} ***************************************'.format(best_val_acc))
                torch.save(net_model.state_dict(), os.path.join('output', model_name + '_' + str(rand_idx) + '_fully.pt'))
        if epoch % args.check_epoch == 0 and epoch != 0:
            test_acc = test(args, net_model, test_loader)
            print('test accuracy:', test_acc, 'epoch=', epoch)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                print('best test accuracy: {} ======================================='.format(best_test_acc))

    print('[best val accuracy]: ', best_val_acc)
    print('[best test accuracy]: ', best_test_acc)


def val(args, net_model, data_loader):
    net_model.eval()

    all_bg_preds = []
    all_evn_preds = []
    all_labels = []

    for i, data in enumerate(data_loader):
        audio_inputs, video_inputs, onehot_labels = data
        audio_inputs = audio_inputs.cuda()
        video_inputs = video_inputs.cuda()
        onehot_labels = onehot_labels.cuda()

        with torch.no_grad():
            bg_preds, evn_preds, _ = net_model(audio_inputs, video_inputs)

        onehot_labels = onehot_labels.cpu().data.numpy().tolist()
        bg_preds = bg_preds.cpu().data.numpy().tolist()
        evn_preds = evn_preds.cpu().data.numpy().tolist()
        all_labels += onehot_labels
        all_bg_preds += bg_preds
        all_evn_preds += evn_preds

    all_labels = np.array(all_labels)
    all_bg_preds = np.array(all_bg_preds)
    all_evn_preds = np.array(all_evn_preds)

    acc = compute_accuracy_supervised(all_bg_preds, all_evn_preds, all_labels, threshold=args.threshold)

    print('[val]acc: ', acc)
    return acc


def test(args, net_model, data_loader, model_path=None):
    if model_path is not None:
        net_model.load_state_dict(torch.load(model_path))
        print(">>> [Testing] Load pretrained model from " + model_path)

    net_model.eval()

    all_bg_preds = []
    all_evn_preds = []
    all_labels = []

    all_fused_feats= []

    for i, data in enumerate(data_loader):
        audio_inputs, video_inputs, onehot_labels = data
        audio_inputs = audio_inputs.cuda()
        video_inputs = video_inputs.cuda()
        onehot_labels = onehot_labels.cuda()

        with torch.no_grad():
            bg_preds, evn_preds, fused_feats = net_model(audio_inputs, video_inputs)

        all_fused_feats += fused_feats.cpu().data.numpy().tolist()

        onehot_labels = onehot_labels.cpu().data.numpy().tolist()
        bg_preds = bg_preds.cpu().data.numpy().tolist()
        evn_preds = evn_preds.cpu().data.numpy().tolist()
        all_labels += onehot_labels
        all_bg_preds += bg_preds
        all_evn_preds += evn_preds

    all_labels = np.array(all_labels)
    all_bg_preds = np.array(all_bg_preds)
    all_evn_preds = np.array(all_evn_preds)

    all_fused_feats = np.array(all_fused_feats)
    np.save('fused_feats', all_fused_feats)

    acc = compute_accuracy_supervised(all_bg_preds, all_evn_preds, all_labels, threshold=args.threshold)

    print('[test]acc: ', acc)

    return acc


if __name__ == "__main__":

    args = parser.parse_args()
    print("args: ", args)

    set_seed(args)

    # model and optimizer
    model_name = args.model_name

    net_model = MAML(a_dim=128, num_layers=args.num_layers, bank_size=args.bank_size, margin=args.margin)

    if args.n_gpus > 1:
        net_model = torch.nn.DataParallel(net_model).cuda()
    else:
        net_model.cuda()

    num_params = count_parameters(net_model)
    print("Total Parameter: \t%2.1fM" % num_params)

    if args.train:
        train(args, net_model)
    else:
        train_loader, val_loader, test_loader = get_loader(args)
        test_acc = test(args, net_model, test_loader, model_path=args.trained_model_path)
        print("[test] accuracy: ", test_acc)
