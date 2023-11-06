"""AVE dataset"""
import numpy as np
import torch
import h5py
import pickle
import random
from itertools import product
import os
import pdb
import torchvision.transforms as T

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

ave_dataset = ['bell', 'Male', 'Bark', 'aircraft', 'car', 'Female', 'Helicopter',
               'Violin', 'Flute', 'Ukulele', 'Fry food', 'Truck', 'Shofar', 'Motorcycle',
               'guitar', 'Train', 'Clock', 'Banjo', 'Goat', 'Baby', 'Bus',
               'Chainsaw', 'Cat', 'Horse', 'Toilet', 'Rodents', 'Accordion', 'Mandolin', 'background']
STANDARD_AVE_DATASET = ['Church bell', 'Male speech, man speaking', 'Bark', 'Fixed-wing aircraft, airplane',
                        'Race car, auto racing', \
                        'Female speech, woman speaking', 'Helicopter', 'Violin, fiddle', 'Flute', 'Ukulele',
                        'Frying (food)', 'Truck', 'Shofar', \
                        'Motorcycle', 'Acoustic guitar', 'Train horn', 'Clock', 'Banjo', 'Goat', 'Baby cry, infant cry',
                        'Bus', 'Chainsaw', \
                        'Cat', 'Horse', 'Toilet flush', 'Rodents, rats, mice', 'Accordion', 'Mandolin']

def get_loader(args, ws=False):
    data_root = './data'
    if not ws:
        train_set = AVEDataset(data_root, 'train')
        val_set = AVEDataset(data_root, 'val')
        test_set = AVEDataset(data_root, 'test')
    else:
        train_set = WSAVEDataset(data_root, 'train')
        val_set = WSAVEDataset(data_root, 'val')
        test_set = WSAVEDataset(data_root, 'test')

    train_sampler = RandomSampler(train_set)
    dev_sampler = SequentialSampler(val_set)
    test_sampler = SequentialSampler(test_set)

    train_loader = DataLoader(train_set,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=2,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_set,
                            sampler=dev_sampler,
                            batch_size=args.eval_batch_size,
                            num_workers=2,
                            pin_memory=False) if val_set is not None else None
    test_loader = DataLoader(test_set,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=2,
                             pin_memory=False) if test_set is not None else None

    return train_loader, val_loader, test_loader


class AVEDataset(Dataset):
    def __init__(self, data_root, split='train'):
        super(AVEDataset, self).__init__()
        self.split = split
        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')
        self.audio_feature_path = os.path.join(data_root, 'audio_feature.h5')
        # self.audio_feature_path = os.path.join(data_root, 'audio_cnn14.h5')
        self.visual_feature_path = os.path.join(data_root, 'visual_feature.h5')
        # Now for the supervised task
        self.labels_path = os.path.join(data_root, 'right_labels.h5')

        with h5py.File(self.sample_order_path, 'r') as hf:
            self.order = hf['order'][:].tolist()  # list
        with h5py.File(self.audio_feature_path, 'r') as hf:
            self.audio_features = torch.from_numpy(hf['avadataset'][:][self.order]).float()  # shape: (?, 10, 128)
            # self.audio_features = torch.from_numpy(hf['dataset'][:][self.order]).float()  # shape: (?, 10, 2048)
        with h5py.File(self.visual_feature_path, 'r') as hf:
            self.video_features = torch.from_numpy(hf['avadataset'][:][self.order]).float()  # shape: (?, 10, 7, 7, 512)
        with h5py.File(self.labels_path, 'r') as hf:
            self.one_hot_labels = torch.from_numpy(hf['avadataset'][:][self.order]).float()  # shape: (?, 10, 29)
        # self.labels = torch.topk(self.one_hot_labels, dim=-1, k=1)[1].squeeze(-1)

        print('>>', split, ' visual feature: ', self.video_features.shape)
        print('>>', split, ' audio feature: ', self.audio_features.shape)

    def __len__(self):
        sample_num = len(self.order)

        return sample_num

    def __getitem__(self, index):
        audio_feat = self.audio_features[index]
        visual_feat = self.video_features[index]
        one_hot_label = self.one_hot_labels[index]
        # label = self.labels[index]

        return audio_feat, visual_feat, one_hot_label


class WSAVEDataset(Dataset):
    # weakly supervised setting
    def __init__(self, data_root, split='train'):
        super(WSAVEDataset, self).__init__()
        self.split = split
        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')
        self.visual_feature_path = os.path.join(data_root, 'visual_feature.h5')
        self.audio_feature_path = os.path.join(data_root, 'audio_feature.h5')
        # self.audio_feature_path = os.path.join(data_root, 'audio_cnn14.h5')

        self.noisy_visual_feature_path = os.path.join(data_root, 'visual_feature_noisy.h5')     # only background
        self.noisy_audio_feature_path = os.path.join(data_root, 'audio_feature_noisy.h5')       # only background
        # self.noisy_audio_feature_path = os.path.join(data_root, 'audio_cnn14_noisy.h5')  # only background

        self.labels_path = os.path.join(data_root, 'right_labels.h5') # original labels only for testing
        # self.dir_labels_path = os.path.join(data_root, 'mil_labels.h5')  # video-level labels
        self.dir_labels_path = os.path.join(data_root, 'prob_label.h5')  # video-level labels
        self.noisy_dir_labels_path = os.path.join(data_root, 'labels_noisy.h5')  # extra labels with only background annotations

        with h5py.File(self.sample_order_path, 'r') as hf:
            self.order = hf['order'][:].tolist()  # list, lenth=3339 (train)
        with h5py.File(self.audio_feature_path, 'r') as hf:
            self.audio_features = torch.from_numpy(hf['avadataset'][:][self.order]).float()  # shape: (?, 10, 128)
            # self.audio_features = torch.from_numpy(hf['dataset'][:][self.order]).float()  # shape: (?, 10, 2048)
        with h5py.File(self.visual_feature_path, 'r') as hf:
            self.video_features = torch.from_numpy(hf['avadataset'][:][self.order]).float()  # shape: (?, 10, 7, 7, 512)

        if split == 'train':
            # extra training data with video-level annotations.
            with h5py.File(self.noisy_audio_feature_path, 'r') as hf:
                extra_audio_features = torch.from_numpy(
                    hf['avadataset'][:]).float()  # shape: (178, 10, 128)
                # extra_audio_features = torch.from_numpy(
                #     hf['dataset'][:]).float()  # shape: (178, 10, 2048)
                extra_audio_features = extra_audio_features[:178]
            with h5py.File(self.noisy_visual_feature_path, 'r') as hf:
                extra_video_features = torch.from_numpy(
                    hf['avadataset'][:]).float()  # shape: (178, 10, 7, 7, 512)
            with h5py.File(self.dir_labels_path, 'r') as hf:
                self.dir_labels = torch.from_numpy(
                    hf['avadataset'][:][self.order]).float()  # (3339, 29)
            with h5py.File(self.noisy_dir_labels_path, 'r') as hf:
                extra_labels = torch.from_numpy(
                    hf['avadataset'][:]).float()  # (178, 29)

            self.audio_features = torch.cat([self.audio_features, extra_audio_features], dim=0)
            self.video_features = torch.cat([self.video_features, extra_video_features], dim=0)
            self.one_hot_labels = torch.cat([self.dir_labels, extra_labels], dim=0)
            # self.labels = torch.topk(self.one_hot_labels, dim=-1, k=1)[1].squeeze(-1)

        else:
            with h5py.File(self.labels_path, 'r') as hf:
                self.one_hot_labels = torch.from_numpy(hf['avadataset'][:][self.order]).float()  # shape: (?, 10, 29)
            # self.labels = torch.topk(self.one_hot_labels, dim=-1, k=1)[1].squeeze(-1)

        print('>>', split, ' visual feature: ', self.video_features.shape)
        print('>>', split, ' audio feature: ', self.audio_features.shape)

    def __len__(self):
        sample_num = len(self.audio_features)
        return sample_num

    def __getitem__(self, index):
        audio_feat = self.audio_features[index]
        visual_feat = self.video_features[index]
        one_hot_label = self.one_hot_labels[index]
        # label = self.labels[index]

        return audio_feat, visual_feat, one_hot_label



