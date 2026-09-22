# CLAPSep

[![arXiv](https://img.shields.io/badge/arXiv-2402.17455-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2402.17455)
[![githubio](https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=Github&style=flat-square)](https://aisaka0v0.github.io/CLAPSep_demo/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/AisakaMikoto/CLAPSep)

Official implementation of CLAPSep: Leveraging Contrastive Pre-trained Models for Multi-Modal Query-Conditioned Target Sound Extraction.

## Data Preparation
Organize raw audio files in AudioCaps as follows:
```
audiocap
├── train
│	├──YzzznDcamMpw.wav
│	├──......
│	└──Y---1_cCGK4M.wav
├── test
│	├──YZYWCwfCkBp4.wav
│	├──......
│	└──Y--0w1YA1Hm4.wav
└── val
	├──YzY3icUyMdh8.wav
	├──......
	└──YrqfQRErjfk8.wav
```

## Training

Modify `./experiments/CLAPSep_base/config.json` according to your local file paths. Run:
```
python train.py ./experiments/CLAPSep_base --use_cuda --gpu_ids 0 1
```

## Evaluation

We have released the metadata for all evaluation benchmarks in `./metadata/evaluation`.

_Prepocessed MUSIC21 dataset can be found [here](https://drive.google.com/file/d/1SYWNWLV_CA_7a77YO5J2mW6XlwVe8Zsl/view?usp=drive_link)._

## Pretrained model

Get the pretrained model on our [huggingface](https://huggingface.co/spaces/AisakaMikoto/CLAPSep/tree/main/model) repo.

## Citation
```
@article{ma2024clapsep,
  title={CLAPSep: Leveraging Contrastive Pre-trained Models for Multi-Modal Query-Conditioned Target Sound Extraction},
  author={Ma, Hao and Peng, Zhiyuan and Li, Xu and Shao, Mingjie and Wu, Xixin and Liu, Ju},
  journal={arXiv preprint arXiv:2402.17455},
  year={2024}
}
```

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE).

### Third-party code and compliance

CLAPSep builds on, and in places adapts, code from third-party projects. The
full attribution notices and reproduced license texts are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

| Component | How it is used | License |
|---|---|---|
| [LAION-CLAP](https://github.com/LAION-AI/CLAP) (`laion-clap`) | Runtime dependency; `model/CLAPSep_decoder.py` adapts `clap_module/htsat.py` building blocks | CC0-1.0 |
| [HTS-AT](https://github.com/RetroCirce/HTS-Audio-Transformer) (Ke Chen) | Carried inside `laion_clap`; adapted via LAION-CLAP | MIT |
| [Microsoft Swin-Transformer](https://github.com/microsoft/Swin-Transformer) | Upstream basis of the HTS-AT layers | MIT |
| [open_clip](https://github.com/mlfoundations/open_clip) | Codebase adopted by LAION-CLAP | MIT |
| [TorchLibrosa](https://github.com/qiuqiangkong/torchlibrosa) | Runtime dependency (STFT/ISTFT/magphase) | MIT |
| [AudioSep](https://github.com/Audio-AGI/AudioSep) | `wav_reconstruct` adapted from `models/resunet.py` | MIT |

All upstream licenses involved are permissive (MIT / CC0-1.0). MIT text is
compatible with both, so redistributing CLAPSep under MIT is clean: the
upstream MIT copyright and permission notices are reproduced in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and CC0-1.0 imposes no
conditions at all.

> **Note on upstream metadata:** the PyPI metadata for `laion-clap` declares the
> classifier "Apache Software License", but the license text actually shipped
> in the package (`laion_clap-<version>.dist-info/licenses/LICENSE`) is
> **CC0 1.0 Universal**. CC0-1.0 is the authoritative license and imposes no
> conditions on reuse.
