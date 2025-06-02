import logging
import torch
import torchmetrics


class MetricMean(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("cumsum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, values: torch.Tensor):
        self.cumsum += torch.sum(values.float())
        self.total += values.numel()

    def compute(self):
        return self.cumsum.float() / (self.total + 1e-10)


class MetricCount(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, values: torch.Tensor):
        self.total += values.numel()

    def compute(self):
        return self.total


class Recorder(torch.nn.Module):
    def __init__(self):
        super(Recorder, self).__init__()
        self.mean = MetricMean()
        self.count = MetricCount()
        self.precision = torchmetrics.Precision()
        self.recall = torchmetrics.Recall()
        self.f1 = torchmetrics.R2Score()
        self.pattern = 'Epoch: {epoch:d}, step: {step:d}, loss: {loss:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, support: {support:d}'

    def forward(self, loss, predictions, labels, **kwargs):
        loss = self.mean(loss)
        count = self.count(predictions)
        f1 = self.f1(predictions, labels)
        precision = self.precision(predictions, labels)
        recall = self.recall(predictions, labels)

        outputs = {
            "loss": loss, "precision": precision, "recall": recall, "f1": f1, "support": count
        }

        return outputs

    def reset(self):
        self.mean.reset()
        self.count.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()

    def _results(self):
        loss = self.mean.compute().detach().cpu()
        count = self.count.compute().detach().cpu()
        precision = self.precision.compute().detach().cpu()
        recall = self.recall.compute().detach().cpu()
        f1 = self.f1.compute().detach().cpu()

        outputs = {
            "loss": loss, "precision": precision, "recall": recall, "f1": f1, "support": count
        }

        return outputs

    def compute(self):
        return self._results()

    def log(self, epoch, num_step, prefix='', suffix=''):
        result = self._results()
        result.update({"epoch": epoch, "step": num_step})
        logging.warning(prefix + self.pattern.format(**result) + suffix)