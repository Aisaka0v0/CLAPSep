# CLAPSep
Official implementation of CLAPSep: Leveraging Contrastive Pre-trained Models for Multi-Modal Query-Conditioned Target Sound Extraction.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/AisakaMikoto/CLAPSep)
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
TODO

We have released the metadata for all evaluation benchmarks in `./metadata/evaluation`.

_Prepocessed MUSIC21 dataset can be found [here](https://drive.google.com/file/d/1SYWNWLV_CA_7a77YO5J2mW6XlwVe8Zsl/view?usp=drive_link)._

## Pretrained model

Get the pretrained model on our [huggingface](https://huggingface.co/spaces/AisakaMikoto/CLAPSep/tree/main/model) repo.
