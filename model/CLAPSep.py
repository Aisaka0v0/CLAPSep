#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Waveformer-main 
@File    ：CLAPSep.py
@IDE     ：PyCharm 
@Author  ：Aisaka/Hao Ma @SDU
@Date    ：2024/2/28 下午1:12 
'''

import torch
import laion_clap
from torchmetrics.audio.snr import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr)
from torchmetrics.audio.sdr import(
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)
import copy
import loralib as lora
from torchlibrosa import ISTFT, STFT, SpecAugmentation
from torchlibrosa.stft import magphase
import librosa
import pytorch_lightning as pl


def loss_fn(pred, tgt):
    return -0.9 * snr(pred, tgt).mean() - 0.1 * si_snr(pred, tgt).mean()


def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def process_model(model, rank):
    for n, module in model.named_modules():
        if 'WindowAttention' in str(type(module)):
            for n_, layer in module.named_modules():
                if isinstance(layer, torch.nn.Linear):
                    lora_layer = lora.Linear(layer.in_features, layer.out_features, r=rank,
                                             bias=hasattr(layer, 'bias'), merge_weights=False)
                    lora_layer.weight = layer.weight
                    if hasattr(layer, 'bias'):
                        lora_layer.bias = layer.bias
                    set_module(model, n+'.'+n_, lora_layer)
    return model


class LightningModule(pl.LightningModule):
    def __init__(self, clap_model, decoder_model, lr, use_lora=False, rank=8, nfft=1024):
        super().__init__()
        self.phase = decoder_model.phase
        self.lr = lr
        self.clap_model = clap_model
        for p in self.clap_model.parameters():
            p.requires_grad = False
        self.audio_branch = copy.deepcopy(self.clap_model.model.audio_branch)
        if use_lora:
            process_model(self.audio_branch, rank)
            lora.mark_only_lora_as_trainable(self.audio_branch, bias='lora_only')

        self.decoder_model = decoder_model
        self.stft = STFT(n_fft=nfft, hop_length=320,
                         win_length=nfft, window='hann', center=True, pad_mode='reflect',
                         freeze_parameters=True)
        self.istft = ISTFT(n_fft=nfft, hop_length=320,
                           win_length=nfft, window='hann', center=True, pad_mode='reflect',
                           freeze_parameters=True)
        self.features = self.install_forward_hooks()

    def training_step(self, batch, batch_idx):
        self.clap_model.eval()
        self.audio_branch.eval()
        mixed, mixed_resample, pos_cap, neg_cap, gt, pos_sample, neg_sample = batch
        real, imag = self.stft(mixed)
        mag, cos, sin = magphase(real, imag)
        with torch.no_grad():
            a = torch.rand((1,)).type_as(gt)
            embed_pos_a, embed_neg_a = torch.chunk(
                self.clap_model.get_audio_embedding_from_data(torch.concat([pos_sample, neg_sample], dim=0),
                                                              use_tensor=True), dim=0, chunks=2)
            embed_pos_t, embed_neg_t = torch.chunk(
                self.clap_model.get_text_embedding(pos_cap + neg_cap, use_tensor=True), dim=0, chunks=2)
            embed_pos = a * embed_pos_a + (1 - a) * embed_pos_t
            embed_neg = a * embed_neg_a + (1 - a) * embed_neg_t
        del self.features[:]
        self.features.append(mag)
        self.audio_branch({"waveform": mixed_resample})
        a = torch.rand((1,))
        if a < 0.25:
            loss = self.cal_loss(embed_pos, torch.zeros_like(embed_pos), mag, cos, sin, length=mixed.size(-1), gt=gt)
        elif a < 0.5:
            loss = self.cal_loss(torch.zeros_like(embed_neg), embed_neg, mag, cos, sin, length=mixed.size(-1), gt=gt)
        else:
            loss = self.cal_loss(embed_pos, embed_neg, mag, cos, sin, length=mixed.size(-1), gt=gt)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, sync_dist=True, batch_size=32)
        del self.features[:]
        return loss

    def cal_loss(self, embed_p, embed_n, mag, cos, sin, length, gt):
        embed = torch.nn.functional.normalize(torch.concat([embed_p, embed_n], dim=-1), dim=-1)
        mask = self.decoder_model(hidden_state=self.features[-1], skip_features=self.features[:-1], embed=embed)
        pred = self.wav_reconstruct(mask, mag, cos, sin, length=length)
        return loss_fn(pred, gt)

    def wav_reconstruct(self, mask, mag_x, cos_x, sin_x, length):
        # ref: https://github.com/Audio-AGI/AudioSep/blob/main/models/resunet.py
        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        if self.phase:
            mag_y = torch.nn.functional.relu_(mag_x * mask[0])
            _, mask_cos, mask_sin = magphase(mask[1], mask[2])
            cos_y = cos_x * mask_cos - sin_x * mask_sin
            sin_y = sin_x * mask_cos + cos_x * mask_sin
        else:
            mag_y = torch.nn.functional.relu_(mag_x * mask)
            cos_y = cos_x
            sin_y = sin_x
        pred = self.istft(mag_y * cos_y, mag_y * sin_y, length=length)
        return pred

    def validation_step(self, batch, batch_idx):
        mixed, mixed_resample, label, neg_label, gt, _, _ = batch
        real, imag = self.stft(mixed)
        mag, cos, sin = magphase(real, imag)
        self.features.append(mag)
        with torch.no_grad():
            embed_pos = self.clap_model.get_text_embedding(label, use_tensor=True)
            embed_neg = self.clap_model.get_text_embedding(neg_label, use_tensor=True)
            embed = torch.concat([embed_pos, embed_neg], dim=-1)
            self.audio_branch({"waveform": mixed_resample})
            mask = self.decoder_model(hidden_state=self.features[-1], skip_features=self.features[:-1], embed=embed)
            pred = self.wav_reconstruct(mask, mag, cos, sin, length=mixed.size(-1))
            loss = si_snr(pred, gt).mean() - si_snr(mixed, gt).mean()
        del self.features[:]
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=64)
        return {"val_loss": loss}

    def on_test_start(self) -> None:
        self.sisdri_vals = torch.tensor([])
        self.sdri_vals = torch.tensor([])

    def test_step(self, batch, batch_idx):
        mixed, mixed_resample, label, neg_label, gt = batch
        real, imag = self.stft(mixed)
        mag, cos, sin = magphase(real, imag)
        with torch.no_grad():
            embed_pos_bached, embed_neg_bached = torch.chunk(self.clap_model.get_text_embedding(label + neg_label, use_tensor=True), chunks=2, dim=0)
            del self.features[:]
            embed = torch.concat([embed_pos_bached, embed_neg_bached], dim=1)
            self.features.append(mag)
            self.audio_branch({"waveform": mixed_resample})
            mask = self.decoder_model(hidden_state=self.features[-1], skip_features=self.features[:-1], embed=embed)
            pred = self.wav_reconstruct(mask, mag, cos, sin, length=mixed.size(-1))
            sisdr = si_sdr(pred, gt).cpu()
            self.sisdri_vals = torch.concat([self.sisdri_vals, sisdr - si_sdr(mixed, gt).cpu()])
            sdr_ = sdr(pred, gt).cpu()
            self.sdri_vals = torch.concat([self.sdri_vals, sdr_ - sdr(mixed, gt).cpu()])
        del self.features[:]
        print('sdri_mean: {}'.format(torch.mean(self.sdri_vals).cpu().numpy()))
        print('sdri_std: {}'.format(torch.std(self.sdri_vals).cpu().numpy()))
        print('sisdri_mean: {}'.format(torch.mean(self.sisdri_vals).cpu().numpy()))
        print('sisdri_std: {}'.format(torch.std(self.sisdri_vals).cpu().numpy()))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=5,
                                                                        verbose=True, min_lr=5e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": schedular,
                "interval": "epoch",
                "monitor": "val_loss"
            },
        }

    def install_forward_hooks(self):
        features = []
        spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                          freq_drop_width=8, freq_stripes_num=2)

        def get_features_list(_, __, output):
            features.append(output)

        def get_features_list_basic_layer(_, __, output):
            features.append(output[0])

        def spec_augmentation_hook(_, __, out):
            out = out.transpose(1, 3)
            out = spec_augmenter(out)
            return out.transpose(1, 3)

        def spectrogram_padding(_, __, out):
            return torch.nn.functional.pad(out, (0, 0, 0, 1024 - out.size(2)))

        self.clap_model.model.audio_branch.bn0.register_forward_hook(spec_augmentation_hook)
        self.audio_branch.spectrogram_extractor.register_forward_hook(spectrogram_padding)
        self.audio_branch.patch_embed.register_forward_hook(get_features_list)
        for module in self.audio_branch.layers:
            module.register_forward_hook(get_features_list_basic_layer)
        return features

    # # this will only save tuned parameters during training
    # def on_save_checkpoint(self, checkpoint):
    #     weights = checkpoint['state_dict']
    #     new_dict = {}
    #     for k, v in weights.items():
    #         if any(e in k for e in ['lora', 'attn.qkv.bias', 'attn.proj.bias', 'decoder_model']):
    #             new_dict[k] = v
    #     checkpoint['state_dict'] = new_dict



