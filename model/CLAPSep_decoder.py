#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Waveformer-main
@File    ：CLAPsep_decoder.py
@IDE     ：PyCharm
@Author  ：Aisaka/Hao Ma @SDU
@Date    ：2023/10/31 下午8:34
'''

from laion_clap.clap_module.htsat import *
from einops import rearrange
import numpy as np

class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * x.sigmoid()


class Glu(nn.Module):

    def __init__(self, dim):
        super(Glu, self).__init__()
        self.dim = dim

    def forward(self, x):
        x_in, x_gate = x.chunk(2, dim=self.dim)
        return x_in * x_gate.sigmoid()


class FiLM(nn.Module):
    def __init__(self, dim_in=1024, hidden_dim=768):
        super(FiLM, self).__init__()
        self.beta = nn.Linear(dim_in, hidden_dim)
        self.gamma = nn.Linear(dim_in, hidden_dim)

    def forward(self, hidden_state, embed):
        embed = embed.unsqueeze(1)
        return self.gamma(embed) * hidden_state + self.beta(embed)


class SkipTrans(nn.Module):
    def __init__(self, in_features, out_features, embed_dim=512, film=True):
        super(SkipTrans, self).__init__()
        self.film = film
        if film:
            self.skip_conv = FiLM(embed_dim, out_features)
        self.feature_proj = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, skip, embed, x=None):
        out = self.feature_proj(skip)
        if self.film:
            out = self.skip_conv(out, embed)
        return self.norm(out) if x is None else self.norm(out + x)

class Conv1d(nn.Conv1d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = "same",
        dilation = 1,
        groups = 1,
        bias = True
    ):
        super(Conv1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros")

        # Assert
        assert padding in ["valid", "same", "causal"]

        # Padding
        if padding == "valid":
            self.pre_padding = None
        elif padding == "same":
            self.pre_padding = nn.ConstantPad1d(padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2), value=0)
        elif padding == "causal":
            self.pre_padding = nn.ConstantPad1d(padding=(kernel_size - 1, 0), value=0)

        # Variational Noise
        self.noise = None
        self.vn_std = None

    def init_vn(self, vn_std):

        # Variational Noise
        self.vn_std = vn_std

    def sample_synaptic_noise(self, distributed):

        # Sample Noise
        self.noise = torch.normal(mean=0.0, std=1.0, size=self.weight.size(), device=self.weight.device, dtype=self.weight.dtype)

        # Broadcast Noise
        if distributed:
            torch.distributed.broadcast(self.noise, 0)

    def forward(self, input):

        # Weight
        weight = self.weight

        # Add Noise
        if self.noise is not None and self.training:
            weight = weight + self.vn_std * self.noise

        # Padding
        if self.pre_padding is not None:
            input = self.pre_padding(input)

        # Apply Weight
        return F.conv1d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ConvolutionModule(nn.Module):
    """Conformer Convolution Module

    Args:
        dim_model: input feature dimension
        dim_expand: output feature dimension
        kernel_size: 1D depthwise convolution kernel size
        Pdrop: residual dropout probability
        stride: 1D depthwise convolution stride
        padding: "valid", "same" or "causal"

    Input: (batch size, input length, dim_model)
    Output: (batch size, output length, dim_expand)

    """

    def __init__(self, dim_model, dim_expand, kernel_size, Pdrop, stride, padding):
        super(ConvolutionModule, self).__init__()

        # Layers
        self.layers = nn.Sequential(
            nn.LayerNorm(dim_model, eps=1e-6),
            Transpose(1, 2),
            Conv1d(dim_model, 2 * dim_expand, kernel_size=1),
            Glu(dim=1),
            Conv1d(dim_expand, dim_expand, kernel_size, stride=stride, padding=padding, groups=dim_expand),
            nn.BatchNorm1d(dim_expand),
            Swish(),
            Conv1d(dim_expand, dim_expand, kernel_size=1),
            Transpose(1, 2),
            nn.Dropout(p=Pdrop)
        )
        self.ln = nn.LayerNorm(dim_expand)

    def forward(self, x):
        return self.ln(self.layers(x)+x)


class BasicLayerDec(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 norm_before_mlp='ln'):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, norm_before_mlp=norm_before_mlp)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample((input_resolution[0]//2, input_resolution[1]//2), dim=dim * 2, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        attns = []
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, attn = blk(x)
                if not self.training:
                    attns.append(attn.unsqueeze(0))
        if not self.training:
            attn = torch.cat(attns, dim = 0)
            attn = torch.mean(attn, dim = 0)
        return x, attn

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        # This is the original implementation in SwinUnet
        # x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)

        # here is our implementation
        # can reverse patch-merging in Swin-Transformer encoder, seems helpful
        x0, x2, x1, x3 = x.chunk(4, dim=-1)
        x = torch.stack((x0, x1, x2, x3), dim=-1)
        x = torch.chunk(x, C // 4, dim=-2)
        x = torch.concat(x, dim=-1).squeeze(-2)
        x = rearrange(x, 'b h w c -> b c h w')
        x = torch.nn.functional.pixel_shuffle(x, 2)
        x = rearrange(x, 'b c h w -> b h w c')
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class InversePatchEmbed(nn.Module):
    """
    Patch Embedding to 2D Image.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True,
                 patch_stride=16):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        padding = ((patch_size[0] - patch_stride[0]) // 2, (patch_size[1] - patch_stride[1]) // 2)

        self.proj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.norm(x)
        if self.flatten:
            # x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            x = x.transpose(1, 2).unflatten(2, self.grid_size).contiguous()  # BNC -> BCHW
        x = self.proj(x)

        return x


class HTSAT_Decoder(nn.Module):
    r"""HTSAT_decoder based on the Swin Transformer
    Args:
        spec_size (int | tuple(int)): Input Spectrogram size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 4
        path_stride (iot | tuple(int)): Patch Stride for Frequency and Time Axis. Default: 4
        in_chans (int): Number of input image channels. Default: 1 (mono)
        num_classes (int): Number of classes for classification head. Default: 527
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each HTSAT-Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, lan_embed_dim=512, spec_size=256, patch_size=4, patch_stride=(4, 4),
                 in_chans=1, num_classes=527,
                 embed_dim=48, depths=[1, 1, 1, 1], num_heads=[4, 8, 16, 32],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False, patch_norm=True,
                 use_checkpoint=False, norm_before_mlp='ln', encoder_embed_dim=96, phase=False,
                 spec_factor=8, d_attn=640, n_masker_layer=4, conv=False):
        super(HTSAT_Decoder, self).__init__()
        self.mel_bins = 64
        self.spec_size = spec_size
        self.phase = phase
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.ape = ape
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = len(self.depths)
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))

        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        self.qkv_bias = qkv_bias
        self.qk_scale = None

        self.patch_norm = patch_norm
        self.norm_layer = norm_layer if self.patch_norm else None
        self.norm_before_mlp = norm_before_mlp
        self.mlp_ratio = mlp_ratio

        self.use_checkpoint = use_checkpoint

        #  process mel-spec ; used only once
        self.freq_ratio = self.spec_size // self.mel_bins


        # split spctrogram into non-overlapping patches
        self.inverse_patch_embed = InversePatchEmbed(
            img_size=self.spec_size, patch_size=self.patch_size, in_chans=self.in_chans,
            embed_dim=self.embed_dim, norm_layer=self.norm_layer, patch_stride=patch_stride)

        patches_resolution = self.inverse_patch_embed.grid_size
        self.patches_resolution = patches_resolution


        # stochastic depth
        dpr = [x.item() for x in
               torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.skip = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayerDec(dim=int(self.embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=self.depths[i_layer],
                               num_heads=self.num_heads[i_layer],
                               window_size=self.window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                               drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                               drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                               norm_layer=self.norm_layer,
                               downsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               norm_before_mlp=self.norm_before_mlp)
            self.layers.append(layer)
            self.skip.append(
                SkipTrans(embed_dim=lan_embed_dim, in_features=int(encoder_embed_dim * 2 ** i_layer), out_features=int(self.embed_dim * 2 ** i_layer)),
            )
        self.layers = self.layers[::-1]
        self.skip = self.skip[::-1]
        # self.skip.append(
        #     SkipTrans(embed_dim=lan_embed_dim, in_features=self.mel_bins, out_features=self.mel_bins),
        # )

        d_spec = self.mel_bins * spec_factor + 1

        self.spec_norm = nn.BatchNorm2d(d_spec, momentum=0.01)
        self.conv = conv
        if not conv:
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_attn, nhead=8,
                                                       dim_feedforward=int(d_attn * self.mlp_ratio),
                                                       batch_first=True, dropout=0)
            transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_masker_layer)

            self.mask_net = nn.Sequential(
                nn.Linear(self.mel_bins + d_spec, d_attn),
                nn.LayerNorm(d_attn),
                transformer_encoder,
                nn.Linear(d_attn, d_spec)
            )
        else:
            self.mask_net = nn.Sequential(
                nn.Linear(self.mel_bins + d_spec, d_spec),
                nn.LayerNorm(d_spec),
                *[ConvolutionModule(dim_model=d_spec, dim_expand=d_spec, kernel_size=9, padding='same',
                                    Pdrop=0, stride=1) for i in range(n_masker_layer)]
            )
        if self.phase:
            self.phase_net = nn.Sequential(
                nn.Linear(self.mel_bins + d_spec, d_spec * 2),
                nn.LayerNorm(d_spec * 2),
                *[ConvolutionModule(dim_model=d_spec * 2, dim_expand=d_spec * 2, kernel_size=9, padding='same',
                                    Pdrop=0, stride=1) for i in range(n_masker_layer)]
            )

        self.film = SkipTrans(embed_dim=lan_embed_dim, in_features=encoder_embed_dim * 8, out_features=self.num_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'absolute_pos_embed'}
    #
    # @torch.jit.ignore
    # def no_weight_decay_keywords(self):
    #     return {'relative_position_bias_table'}

    def forward(self, hidden_state, skip_features, embed):
        skip_features = skip_features[::-1]
        # hidden_state = torch.randn(hidden_state.shape).type_as(hidden_state)

        spec = skip_features[-1]

        h = self.film(hidden_state, embed)

        for i, (layer, f, skip) in enumerate(zip(self.layers, skip_features, self.skip)):
            h = layer(h)[0]
            h = skip(skip=f, embed=embed, x=h)

        h = self.reshape_img2wav(self.inverse_patch_embed(h)).squeeze(1)

        h = h[:, :spec.size(2), :]

        spec = spec.transpose(1, 3)

        spec = self.spec_norm(spec).transpose(1, 3).squeeze(1)

        h = torch.concat([spec, h], dim=-1)

        mask = self.mask_net(h).unsqueeze(1)

        if self.phase:
            mask_r, mask_i = torch.chunk(self.phase_net(h).unsqueeze(1), chunks=2, dim=-1)
            return torch.sigmoid(mask), torch.tanh(mask_r), torch.tanh(mask_i)
        else:
            return torch.sigmoid(mask)

    def reshape_img2wav(self, x):
        # (B, 1, 256, 256)
        x = x.reshape(x.shape[0], x.shape[1], self.freq_ratio, x.shape[2]//self.freq_ratio, x.shape[3]) # (B, 1, 4, 64, 256)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3] * x.shape[4])
        x = x.permute(0, 1, 3, 2).contiguous()
        return x


# if __name__ == "__main__":
#     import torch
#     from msclap import CLAP
#     import os
#     import torchaudio
#     import torchaudio.transforms as T
#     import numpy as np
#     import random
#     from torchlibrosa import Spectrogram, LogmelFilterBank
#     clap_model = CLAP(model_fp="/home/user/202212661/clapsep/Waveformer-main/checkpoint_path/CLAP_weights_2023.pth",
#                       version='2023', use_cuda=True)
#     text_data = [
#         "Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation",
#         "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard",
#         "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano",
#         "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel",
#         "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling",
#         "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe",
#         "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak",
#         "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle",
#         "Writing"]
#     # Extract text embeddings
#     text_embeddings = clap_model.get_text_embeddings(text_data)
#     path = "/home/user/202212661/clapsep/Waveformer-main/data/FSDSoundScapes/FSDKaggle2018/train/Tearing/2232ce13.wav"
#     # Extract audio embeddings
#     audio_embeddings_ = clap_model.get_audio_embeddings([path])
#
#     window = 'hann'
#     center = True
#     pad_mode = 'reflect'
#     ref = 1.0
#     amin = 1e-10
#     top_db = None
#
#     spectrogram_extractor = Spectrogram(n_fft=512, hop_length=160,
#                                         win_length=512, window=window, center=center, pad_mode=pad_mode,
#                                         freeze_parameters=True).cuda()
#     # Logmel feature extractor
#     logmel_extractor = LogmelFilterBank(sr=16000, n_fft=512,
#                                         n_mels=64, fmin=0, fmax=8000, ref=ref, amin=amin,
#                                         top_db=top_db,
#                                         freeze_parameters=True).cuda()
#
#     clap_model.clap.audio_encoder.base.htsat.spectrogram_extractor = spectrogram_extractor
#     clap_model.clap.audio_encoder.base.htsat.logmel_extractor = logmel_extractor
#
#     features = []
#
#
#     def get_features_list(module, input, output):
#         features.append(output)
#
#
#     def get_features_list_basic_layer(module, input, output):
#         features.append(output[0])
#
#
#     clap_model.clap.audio_encoder.base.htsat.patch_embed.register_forward_hook(get_features_list)
#     for module in clap_model.clap.audio_encoder.base.htsat.layers:
#         module.register_forward_hook(get_features_list_basic_layer)
#
#     audio_time_series, sample_rate = torchaudio.load(path)
#     resample_rate = 16000
#     if resample_rate != sample_rate:
#         resampler = T.Resample(sample_rate, resample_rate)
#         audio_time_series = resampler(audio_time_series)
#
#     sample_rate = resample_rate
#     audio_duration = 10
#     audio_time_series = audio_time_series.reshape(-1)
#     if audio_duration * sample_rate >= audio_time_series.shape[0]:
#         repeat_factor = int(np.ceil((audio_duration * sample_rate) /
#                                     audio_time_series.shape[0]))
#         # Repeat audio_time_series by repeat_factor to match audio_duration
#         audio_time_series = audio_time_series.repeat(repeat_factor)
#         # remove excess part of audio_time_series
#         audio_time_series = audio_time_series[0:audio_duration * sample_rate]
#     else:
#         # audio_time_series is longer than predefined audio duration,
#         # so audio_time_series is trimmed
#         start_index = random.randrange(
#             audio_time_series.shape[0] - audio_duration * sample_rate)
#         audio_time_series = audio_time_series[start_index:start_index +
#                                                           audio_duration * sample_rate]
#
