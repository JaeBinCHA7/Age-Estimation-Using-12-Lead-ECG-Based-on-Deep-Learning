import os
import pandas as pd
import time
import torch
from sklearn.model_selection import train_test_split
import numpy as np


######################################################################################################################
#                                                   for dataset                                                      #
######################################################################################################################
def process_files(opt, directory_path, csv_path):
    all_files = os.listdir(directory_path)
    npy_files = [f for f in all_files if f.endswith('.npy')]

    df = pd.read_csv(csv_path)
    matched_rows = []

    for npy_file in npy_files:
        match = df[df["FILENAME"] == npy_file[:-4]]  # 확장자 제외
        if not match.empty:
            for _, row in match.iterrows():
                matched_rows.append({
                    'npy_path': os.path.join(directory_path, npy_file),
                    'Filename': row['FILENAME'],
                    'Gender': row['GENDER'],
                    'Age': row['AGE']
                })

    matched_df = pd.DataFrame(matched_rows)

    if opt.mode == 'child':
        bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Adjust as necessary
        matched_df['age_bins'] = pd.cut(matched_df['Age'], bins=bins, right=False)
        # na_rows = matched_df[matched_df.isna().any(axis=1)]
    elif opt.mode == 'adult':
        bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
        matched_df['age_bins'] = pd.cut(matched_df['Age'], bins=bins, right=False)

    train_df, valid_df = train_test_split(matched_df, test_size=0.1, random_state=42, stratify=matched_df['age_bins'])

    train_df = train_df.drop(columns=['age_bins'])
    valid_df = valid_df.drop(columns=['age_bins'])

    return train_df, valid_df


# make a new dir
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


######################################################################################################################
#                                                   for training                                                     #
######################################################################################################################
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


# calculate the size of total network
def cal_total_params(our_model):
    total_parameters = 0
    for variable in our_model.parameters():
        shape = variable.size()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    return total_parameters


class Bar(object):
    def __init__(self, dataloader):
        if not hasattr(dataloader, 'dataset'):
            raise ValueError('Attribute `dataset` not exists in dataloder.')
        if not hasattr(dataloader, 'batch_size'):
            raise ValueError('Attribute `batch_size` not exists in dataloder.')

        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self._idx = 0
        self._batch_idx = 0
        self._time = []
        self._DISPLAY_LENGTH = 50

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._time) < 2:
            self._time.append(time.time())

        self._batch_idx += self.batch_size
        if self._batch_idx > len(self.dataset):
            self._batch_idx = len(self.dataset)

        try:
            batch = next(self.iterator)
            self._display()
        except StopIteration:
            raise StopIteration()

        self._idx += 1
        if self._idx >= len(self.dataloader):
            self._reset()

        return batch

    def _display(self):
        if len(self._time) > 1:
            t = (self._time[-1] - self._time[-2])
            eta = t * (len(self.dataloader) - self._idx)
        else:
            eta = 0

        rate = self._idx / len(self.dataloader)
        len_bar = int(rate * self._DISPLAY_LENGTH)
        bar = ('=' * len_bar + '>').ljust(self._DISPLAY_LENGTH, '.')
        idx = str(self._batch_idx).rjust(len(str(len(self.dataset))), ' ')

        tmpl = '\r{}/{}: [{}] - ETA {:.1f}s'.format(
            idx,
            len(self.dataset),
            bar,
            eta
        )
        print(tmpl, end='')
        if self._batch_idx == len(self.dataset):
            print()

    def _reset(self):
        self._idx = 0
        self._batch_idx = 0
        self._time = []
