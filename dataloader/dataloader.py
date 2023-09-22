import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils import process_files
import numpy as np
from scipy.signal import butter, filtfilt


def create_dataloader(opt):
    print('Load the dataset...')

    if opt.mode == 'child':
        if not os.path.exists('./dataset/train_dataset_child.csv'):
            print('There is no dataset')
            print('Make new dataset')
            train_df, valid_df = process_files(opt, opt.train_for_child, opt.train_for_child_label)

            train_df.to_csv('./dataset/train_dataset_child.csv')
            valid_df.to_csv('./dataset/valid_dataset_child.csv')

            train = pd.DataFrame(train_df)  # [:500]
            valid = pd.DataFrame(valid_df)  # [:500]
            print(len(train))
        else:
            train_header = pd.read_csv('./dataset/train_dataset_child.csv')
            valid_header = pd.read_csv('./dataset/valid_dataset_child.csv')
            train = pd.DataFrame(train_header)  # [:500]
            valid = pd.DataFrame(valid_header)  # [:100]

            print(len(train))
    elif opt.mode == 'adult':
        if not os.path.exists('./dataset/train_dataset_adult.csv'):
            print('There is no dataset')
            print('Make new dataset')
            train_df, valid_df = process_files(opt, opt.train_for_adult, opt.train_for_adult_label)

            train_df.to_csv('./dataset/train_dataset_adult.csv')
            valid_df.to_csv('./dataset/valid_dataset_adult.csv')

            train = pd.DataFrame(train_df)  # [:500]
            valid = pd.DataFrame(valid_df)  # [:500]
            print(len(train))
        else:
            train_header = pd.read_csv('./dataset/train_dataset_adult.csv')
            valid_header = pd.read_csv('./dataset/valid_dataset_adult.csv')
            train = pd.DataFrame(train_header)  # [:500]
            valid = pd.DataFrame(valid_header)  # [:100]

            print(len(train))

    train_loader = DataLoader(
        dataset=ECG_Dataset(opt, train, mode='train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        sampler=None
    )

    validatioin_loader = DataLoader(
        dataset=ECG_Dataset(opt, valid, mode='valid'),
        batch_size=opt.batch_size, shuffle=False, num_workers=0, drop_last=True
    )

    return train_loader, validatioin_loader


def create_dataloader_test(opt):
    adult_files = os.listdir(opt.test_for_adult)
    adult_files = [os.path.join(opt.test_for_adult, f) for f in adult_files if f.endswith('.npy')]

    child_files = os.listdir(opt.test_for_child)
    child_files = [os.path.join(opt.test_for_child, f) for f in child_files if f.endswith('.npy')]

    test_loader_adult = DataLoader(
        dataset=ECG_Dataset_Test(opt, adult_files, mode='test'),
        batch_size=1, shuffle=False, num_workers=0
    )

    test_loader_child = DataLoader(
        dataset=ECG_Dataset_Test(opt, child_files, mode='test'),
        batch_size=1, shuffle=False, num_workers=0
    )

    return test_loader_adult, test_loader_child


def normalize_ecg(ecg_data):
    """Z-score normalize ECG data along the time dimension of each channel.

    Args:
        ecg_data (np.ndarray): The ECG data. Shape: (n_channels, n_timepoints)

    Returns:
        np.ndarray: The normalized ECG data.
    """
    mean = np.mean(ecg_data, axis=1, keepdims=True)
    std = np.std(ecg_data, axis=1, keepdims=True)
    return (ecg_data - mean) / (std + 1e-8)  # Small constant to prevent division by zero


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    """Butterworth bandpass filter.

    Args:
        data (np.ndarray): The ECG data. Shape: (n_channels, n_timepoints)
        lowcut (float): Low cut-off frequency
        highcut (float): High cut-off frequency
        fs (int): The sampling rate of the data
        order (int): The order of the filter. Default is 3.

    Returns:
        np.ndarray: The filtered ECG data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    filtered_data = filtfilt(b, a, data, axis=1)
    return filtered_data


class ECG_Dataset(Dataset):
    def __init__(self, opt, dataset, mode):
        # load data
        self.fs = opt.fs
        self.samples = opt.samples
        self.mode = mode
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        targets = self.dataset.iloc[idx]['Age']

        inputs = np.load((self.dataset.iloc[idx]['npy_path'])).reshape(12, 5000)
        inputs = inputs[:, :4096]
        inputs = normalize_ecg(inputs)
        inputs = butter_bandpass_filter(inputs, lowcut=1, highcut=47, fs=self.fs)

        inputs = np.nan_to_num(inputs)
        inputs = torch.from_numpy(inputs)

        return inputs, targets


class ECG_Dataset_Test(Dataset):
    def __init__(self, opt, dataset, mode):
        self.fs = opt.fs
        self.samples = opt.samples
        self.mode = mode
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        file_name_with_ext = self.dataset[idx].split('/')[-1]  # 'ecg_adult_38019.npy'
        file_name = file_name_with_ext.rsplit('.', 1)[0]  # 'ecg_adult_38019'

        inputs = np.load(self.dataset[idx]).reshape(12, 5000)
        inputs = inputs[:, :4096]
        inputs = normalize_ecg(inputs)
        inputs = butter_bandpass_filter(inputs, lowcut=1, highcut=47, fs=self.fs)

        inputs = np.nan_to_num(inputs)
        inputs = torch.from_numpy(inputs)

        return inputs, file_name
