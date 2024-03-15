import os
import random

import torch
import torchaudio
import torchaudio.transforms as AT
import csv
import numpy as np
import librosa
import pandas as pd
import laion_clap
from model.CLAPSep import LightningModule
from model.CLAPSep_decoder import HTSAT_Decoder
import argparse
import pytorch_lightning as pl
from helpers import utils as local_utils


class AudioCapsTest(torch.utils.data.Dataset):  # type: ignore

    def __init__(self, input_dir, sr=32000,
                 resample_rate=48000):
        self.data_path = input_dir

        self.data_names = []
        self.data_caps = []
        self.noise_names = []
        self.noise_caps = []
        with open('../metadata/evaluation/audiocaps_eval.csv') as d:
            reader = csv.reader(d, skipinitialspace=True)
            next(reader)
            for row in reader:
                self.data_names.append(row[0])
                self.data_caps.append(row[1])
                self.noise_names.append(row[2])
                self.noise_caps.append(row[3])

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
            # audio = torch.nn.functional.pad(audio, (0, self.max_length - audio.size(1)), 'constant')
            wav = np.pad(wav, (0, max_length - len(wav)), 'constant')
        return wav

    def __getitem__(self, idx):

        tgt_name = self.data_names[idx]
        noise_name = self.noise_names[idx]
        tgt_cap = self.data_caps[idx]
        neg_cap = self.noise_caps[idx]

        assert noise_name != tgt_name
        snr = torch.ones((1,)) * 0
        tgt = torch.tensor(self.load_wav(os.path.join(self.data_path, tgt_name))).unsqueeze(0)
        noise = torch.tensor(self.load_wav(os.path.join(self.data_path, noise_name))).unsqueeze(0)
        mixed = torchaudio.functional.add_noise(tgt, noise, snr=snr)

        max_value = torch.max(torch.abs(mixed))
        if max_value > 1:
            tgt *= 0.9 / max_value
            mixed *= 0.9 / max_value

        tgt = tgt.squeeze()
        mixed = mixed.squeeze()

        return mixed, self.resampler(mixed), tgt_cap, neg_cap, tgt



def main(args):
    torch.set_float32_matmul_precision('highest')
    # Load dataset
    data_test = AudioCapsTest(input_dir='/home/user/202212661/clapsep/Waveformer-main/data/audiocap/test')

    test_loader = torch.utils.data.DataLoader(data_test,
                                             batch_size=8,
                                             num_workers=16,
                                             pin_memory=True,
                                             shuffle=False)

    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device='cpu')
    clap_model.load_ckpt(args.clap_path)
    decoder = HTSAT_Decoder(**args.model)
    lightning_module = LightningModule(clap_model, decoder, lr=args.optim['lr'],
                                       use_lora=args.lora,
                                       rank=args.lora_rank,
                                       nfft=args.nfft)
    distributed_backend = "ddp"
    trainer = pl.Trainer(
        default_root_dir=os.path.join(args.exp_dir, 'checkpoint'),
        devices=args.gpu_ids if args.use_cuda else "auto",
        accelerator="gpu" if args.use_cuda else "cpu",
        benchmark=False,
        gradient_clip_val=5.0,
        precision='bf16-mixed',
        limit_train_batches=1.0,
        max_epochs=args.epochs,
        strategy=distributed_backend,
    )

    # weight = torch.load("/home/user/202212661/clapsep/Waveformer-main/clap/src/model/best_model.ckpt")
    # lightning_module.load_state_dict(weight, strict=False)

    trainer.test(model=lightning_module, dataloaders=test_loader,
                ckpt_path='/home/user/202212661/clapsep/Waveformer-main/experiments/laion_simple_backend/checkpoint/lightning_logs/version_639/checkpoints/epoch=149-step=231000.ckpt')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data Params
    parser.add_argument('exp_dir', type=str,
                        default='/home/user/202212661/clapsep/Waveformer-main/experiments/sepformer',
                        help="Path to save checkpoints and logs.")

    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                        help="Whether to use cuda")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None,
                        help="List of GPU ids used for training. "
                             "Eg., --gpu_ids 2 4. All GPUs are used by default.")

    args = parser.parse_args()

    # Set the random seed for reproducible experiments
    pl.seed_everything(114514)
    # Set up checkpoints
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # Load model and training params
    params = local_utils.Params(os.path.join(args.exp_dir, 'config.json'))
    for k, v in params.__dict__.items():
        vars(args)[k] = v
    main(args)