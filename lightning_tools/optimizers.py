from transformers import (
    AdamW,
    # get_linear_schedule_with_warmup
)
import functools
from torch.optim.lr_scheduler import LambdaLR
from lightning_tools.radam import RAdam
import torch

def lr_lambda(current_step, num_warmup_steps, num_training_steps):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
    )


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    return LambdaLR(optimizer, functools.partial(lr_lambda, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps), last_epoch)


def config_nodecay_optimizer(model, init_lr, warmup_steps, max_steps, name='bert_lr', weight_decay=0.01):
    """
    Original Bert Optimizer do not decay for bias and layer_normal
    Args:
        model:
        init_lr:
        warmup_steps:
        max_steps:
        name:
        weight_decay:

    Returns:

    """
    no_decay = ['bias', "LayerNorm", "layer_norm"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            # Whether or not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
            "correct_bias": False,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "correct_bias": False
        }
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=init_lr, eps=1e-8, correct_bias=False
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
    )
    scheduler = {'scheduler': scheduler, 'name': name, 'interval': 'step', 'frequency': 1}

    return optimizer, scheduler


def config_optimizer(parameters, init_lr, warmup_steps, max_steps, name='lr'):
    """
    Original Bert Optimizer do not decay for bias and layer_normal
    Args:
        parameters:
        init_lr:
        warmup_steps:
        max_steps:
        name:
        weight_decay:

    Returns:

    """
    optimizer = AdamW(
        parameters, lr=init_lr, eps=1e-8, correct_bias=False
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
    )
    scheduler = {'scheduler': scheduler, 'name': name, 'interval': 'step', 'frequency': 1}

    return optimizer, scheduler


class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        model_size,
        warmup,
        last_epoch=-1,
    ):

        self.warmup = warmup
        self.factor = 1.0
        self.model_size = model_size
        super(NoamLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.factor * \
            (self.model_size ** (-0.5) *
            min((self.last_epoch + 1) ** (-0.5), (self.last_epoch + 1) * self.warmup ** (-1.5)))
            for base_lr in self.base_lrs
        ]


def config_decoder_optimizer(parameters, init_lr, warmup_steps, model_size, name='decoder_lr'):
    optimizer = RAdam(
        parameters,
        lr=init_lr
    )

    scheduler = NoamLR(optimizer, model_size=model_size, warmup=warmup_steps)
    scheduler = {'scheduler': scheduler, 'name': name, 'interval': 'step', 'frequency': 1}

    return optimizer, scheduler



