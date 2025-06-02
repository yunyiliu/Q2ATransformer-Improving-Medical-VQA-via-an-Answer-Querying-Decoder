import os
import logging
from .csv_logger import CsvLogger
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from pytorch_lightning.plugins.environments import ClusterEnvironment


def add_callbacks(args):
    # time for current runs
    # timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # log_dir = os.path.join(args.savedmodel_path, timestamp)

    log_dir = args.savedmodel_path
    os.makedirs(log_dir, exist_ok=True)

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:  logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(log_dir, "val_log.txt"),
                        filemode='w+')

    # --------- Add Callbacks
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=os.path.join(log_dir, "checkpoints"),
    #     filename="{epoch}-{step}-{bleu:.4f}-{cider:.4f}",
    #     save_top_k=-1,
    #     every_n_epochs=0,
    #     every_n_train_steps=0,
    #     save_last=True
    # )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename="{best_model}",
        save_top_k=-1,
        every_n_epochs=0,
        every_n_train_steps=0,
        save_last=True
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(log_dir, "logs"), name="tensorboard")
    csv_logger = CsvLogger(save_dir=os.path.join(log_dir, "logs"), name="csvlog")

    to_returns = {
        "callbacks": [checkpoint_callback, lr_monitor_callback],
        "loggers": [csv_logger, tb_logger]
    }
    return to_returns


class CustomClusterEnvironment(ClusterEnvironment):
    def creates_children(self) -> bool:
        # return True if the cluster is managed (you don't launch processes yourself)
        return True

    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])

    def global_rank(self) -> int:
        return int(os.environ["RANK"])

    def local_rank(self) -> int:
        return int(os.environ["LOCAL_RANK"])

    def node_rank(self) -> int:
        return int(os.environ["NODE_RANK"])

    def master_address(self) -> str:
        return os.environ["MASTER_ADDRESS"]

    def master_port(self) -> int:
        return int(os.environ["MASTER_PORT"])

    def set_global_rank(self, rank: int) -> None:
        pass

    def set_world_size(self, size: int) -> None:
        pass
