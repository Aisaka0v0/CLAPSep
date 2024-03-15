#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Waveformer-main
@File    ：dataset_online.py
@IDE     ：PyCharm
@Author  ：Aisaka/Hao Ma @SDU
@Date    ：2023/11/1 下午6:47
'''
import os
import random

import torch
import torchaudio
import torchaudio.transforms as AT
import csv
import numpy as np
import librosa


class AudioCapMix(torch.utils.data.Dataset):  # type: ignore

    def __init__(self, input_dir, dset='', sr=None,
                 resample_rate=None):
        assert dset in ['train', 'val'], \
            "`dset` must be one of ['train', 'val']"
        self.dset = dset
        self.data_path = os.path.join(input_dir, dset)
        self.data_meta = dict()
        with open(os.path.join('./metadata/training', dset + '.csv'), encoding='utf-8') as d:
            reader = csv.reader(d, skipinitialspace=True)
            for row in reader:
                self.data_meta[row[0]] = row[1]
        self.data_meta.pop('file_name')
        self.augmentation = torchaudio.transforms.SpeedPerturbation(48000, [0.9, 1.1])

        self.data_names = list(self.data_meta.keys())
        if dset == 'val':
            self.noise_names = []
            for name in self.data_names:
                while True:
                    noise_name = random.sample(self.data_names, 1)[0]
                    if noise_name != name:
                        break
                self.noise_names.append(noise_name)

        if resample_rate is not None:
            self.resampler = AT.Resample(sr, resample_rate)
            self.sr = sr
            self.resample_rate = resample_rate
        else:
            self.sr = sr

    def __len__(self):
        return len(self.data_names)

    def load_wav(self, path):
        max_length = self.sr * 10
        wav = librosa.core.load(path, sr=self.sr)[0]
        if len(wav) > max_length:
            wav = wav[0:max_length]

        # pad audio to max length, 10s for AudioCaps
        if len(wav) < max_length:
            wav = np.pad(wav, (0, max_length - len(wav)), 'constant')
        return wav

    def __getitem__(self, idx):

        tgt_name = self.data_names[idx]
        if self.dset =='train':
            while True:
                noise_name = random.sample(self.data_names, 1)[0]
                if noise_name != tgt_name:
                    break
        else:
            noise_name = self.noise_names[idx]

        snr = torch.zeros((1,))
        # snr = (torch.rand((1,)) * 10 - 5) if self.dset == 'train' else torch.zeros((1,))
        tgt = torch.tensor(self.load_wav(os.path.join(self.data_path, tgt_name))).unsqueeze(0)
        noise = torch.tensor(self.load_wav(os.path.join(self.data_path, noise_name))).unsqueeze(0)
        mixed = torchaudio.functional.add_noise(tgt, noise, snr=snr)
        pos_sample, _ = self.augmentation(self.resampler(tgt.squeeze()))
        neg_sample, _ = self.augmentation(self.resampler(noise.squeeze()))

        max_value = torch.max(torch.abs(mixed))
        if max_value > 1:
            tgt *= 0.9 / max_value
            mixed *= 0.9 / max_value

        tgt = tgt.squeeze()
        mixed = mixed.squeeze()
        tgt_cap = self.data_meta[tgt_name]
        neg_cap = self.data_meta[noise_name]

        mixed_resample = self.resampler(mixed)

        return mixed, mixed_resample, tgt_cap, neg_cap, tgt, self.pad_or_trim(pos_sample), self.pad_or_trim(neg_sample)

    def pad_or_trim(self, wav_in):
        target_len = 48000*10
        if wav_in.size(0) < target_len:
            wav_in = torch.nn.functional.pad(wav_in, (0, target_len-wav_in.size(0)))
        elif wav_in.size(0) > target_len:
            wav_in = wav_in[:target_len]
        max_value = torch.max(torch.abs(wav_in))
        if max_value > 1:
            wav_in *= 0.9 / max_value
        return wav_in


# if __name__ == "__main__":
#     dataset = FSDSoundScapesDataset(input_dir="/home/user/202212661/clapsep/Waveformer-main/data/audiocap",
#         dset="test",
#         sr=32000,
#         resample_rate=48000,
#         return_neg=True)
#     mixed, tgt_cap, tgt = dataset.__getitem__(1)
#     print()