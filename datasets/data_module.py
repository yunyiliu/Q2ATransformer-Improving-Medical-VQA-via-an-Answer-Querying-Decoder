import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from datasets.data_helper_RAD import FieldParser, create_datasets
from pytorch_lightning.trainer.supporters import CombinedLoader

AVAIL_GPUS = min(1, torch.cuda.device_count())


class DataModule(LightningDataModule):

    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args

        if args.test_mode:
            train_dataset, dev_dataset, test_dataset = create_datasets(self.args)

            self.dataset = {
                "train": train_dataset, "validation": dev_dataset, "test": test_dataset
            }


    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.

        download

        tokenize

        etc…
        :return:
        """

    def setup(self, stage: str):
        """
        There are also data operations you might want to perform on every GPU. Use setup to do things like:

        count number of classes

        build vocabulary

        perform train/val/test splits

        apply transforms (defined explicitly in your datamodule or assigned in init)

        etc…
        :param stage:
        :return:
        """
        train_dataset, dev_dataset, test_dataset = create_datasets(self.args)
        self.dataset = {
            "train": train_dataset, "validation": dev_dataset, "test": test_dataset
        }
        

    def train_dataloader(self):
        """
        Use this method to generate the train dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """

        return DataLoader(self.dataset["train"], batch_size=self.args.batch_size, drop_last=True, pin_memory=True,
                          num_workers=self.args.cpu_num)


    def val_dataloader(self):
        """
        Use this method to generate the val dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        loader = DataLoader(self.dataset["validation"], batch_size=self.args.val_batch_size, drop_last=False, pin_memory=False,
                            num_workers=self.args.cpu_num)
        return loader


    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.args.val_batch_size, drop_last=False, pin_memory=False,
                            num_workers=self.args.cpu_num)

