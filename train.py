import os
import logging
import torch
import pytorch_lightning as pl
import laion_clap
from model.CLAPSep_decoder import HTSAT_Decoder
from model.CLAPSep import LightningModule
import argparse
from helpers import utils as local_utils
from data_utils.audiocap_dataset import AudioCapMix as Dataset


def main(args):
    torch.set_float32_matmul_precision('medium')
    # Load dataset
    data_train = Dataset(**args.train_data)
    logging.info("Loaded train dataset at %s containing %d elements" %
                 (args.train_data['input_dir'], len(data_train)))
    data_val = Dataset(**args.val_data)
    logging.info("Loaded test dataset at %s containing %d elements" %
                 (args.val_data['input_dir'], len(data_val)))
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.n_workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(data_val,
                                             batch_size=args.eval_batch_size,
                                             num_workers=args.n_workers,
                                             pin_memory=True)

    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device='cpu')
    clap_model.load_ckpt(args.clap_path)
    decoder = HTSAT_Decoder(**args.model)
    lightning_module = LightningModule(clap_model, decoder, lr=args.optim['lr'],
                                       use_lora=args.lora,
                                       rank=args.lora_rank,
                                       nfft=args.nfft,)
    # distributed_backend = "ddp_find_unused_parameters_true"
    distributed_backend = "ddp"
    trainer = pl.Trainer(
        default_root_dir=os.path.join(args.exp_dir, 'checkpoint'),
        devices=args.gpu_ids if args.use_cuda else "auto",
        accelerator="gpu" if args.use_cuda else "cpu",
        benchmark=True,
        gradient_clip_val=5.0,
        precision='bf16-mixed',
        limit_train_batches=1.0,
        max_epochs=args.epochs,
        strategy=distributed_backend,
    )

    trainer.fit(model=lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader,)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data Params
    parser.add_argument('exp_dir', type=str,
                        default='./experiments/CLAPSep_base',
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



