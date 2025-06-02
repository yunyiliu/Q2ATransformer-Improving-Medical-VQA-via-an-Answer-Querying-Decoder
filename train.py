import os
from pprint import pprint
import pytorch_lightning as pl
from configs.config import parser
from datasets.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
import torch
import matplotlib.pyplot as plt
from models.q2a_transformer import MultiModal
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train(args):
    dm = DataModule(args)
    callbacks = add_callbacks(args)
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks["callbacks"], logger=callbacks["loggers"]
    )
    # model = MultiModal.load_from_checkpoint("/root/VQA_Main/Modified_MedVQA-main/sa/best_model=best_model.ckpt", hparams_file=args.hparams_file, strict=False)

    if args.ckpt_file is not None:
        model = MultiModal.load_from_checkpoint(args.ckpt_file, hparams_file=args.hparams_file, strict=False)
    else:
        model = MultiModal(args)
    trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))

    train(args)

if __name__ == '__main__':
    main()