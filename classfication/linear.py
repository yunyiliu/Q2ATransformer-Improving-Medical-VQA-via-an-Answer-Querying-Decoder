import torch.nn as nn
import torch
import torch.nn.functional as F
import torchmetrics



class Dice(nn.Module):
    def __init__(self, num_features, dim=2):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3
        self.bn = nn.BatchNorm1d(num_features, eps=1e-9)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
        self.num_features = num_features

    def forward(self, x):
        if self.dim == 3:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            alpha = torch.zeros((self.num_features, 1), device=x.device)
            out = alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)

        elif self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            alpha = torch.zeros((self.num_features,), device=x.device)
            out = alpha * (1 - x_p) * x + x_p * x
        else:
            raise NotImplementedError

        return out


class Projector(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size, dropout=0.2):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(input_size, intermediate_size)
        self.bn1 = nn.BatchNorm1d(intermediate_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_size, output_size)
        # self.ln = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.fc2(x)
        # x = self.ln(x)

        return x


class LangOutput(nn.Module):

    def __init__(self, input_size, intermediate_size, output_size, dropout=0.2):
        super(LangOutput, self).__init__()
        self.projector_layer = Projector(input_size, intermediate_size, output_size, dropout)

    def forward(self, inputs):

        return self.projector_layer(inputs)


class VisionOutput(nn.Module):

    def __init__(self, input_size, intermediate_size, output_size, dropout=0.2):
        super(VisionOutput, self).__init__()
        self.projector_layer = Projector(input_size, intermediate_size, output_size, dropout)

    def forward(self, inputs):
        return self.projector_layer(inputs)


class TagOutput(nn.Module):

    def __init__(self, input_size, intermediate_size, output_size, dropout=0.2):
        super(TagOutput, self).__init__()
        self.projector_layer = Projector(input_size, intermediate_size, output_size, dropout)

    def forward(self, inputs):
        return self.projector_layer(inputs)


class AutoEncoderOutput(nn.Module):

    def __init__(self, input_size, intermediate_size, output_size, dropout=0.2):
        super(AutoEncoderOutput, self).__init__()
        self.in_projector_layer = Projector(input_size, intermediate_size, output_size, dropout)
        self.out_projector_layer = Projector(output_size, intermediate_size, input_size, dropout)

    def forward(self, inputs):

        return self.in_projector_layer(inputs)

    def forward_avg(self, states, split=4):
        """

        Args:
            states:
            split: 4 type embed

        Returns:

        """
        outputs = self.out_projector_layer(states)

        split_size = int(outputs.size(-1) / split)
        avg_pool = torch.cat([x.unsqueeze(1) for x in torch.split(outputs, split_size, dim=-1)], dim=1).mean(1)

        return avg_pool

    def compute_loss(self, inputs):
        states = self.in_projector_layer(inputs)
        outputs = self.out_projector_layer(states)

        loss = F.mse_loss(inputs, outputs)

        return loss


class ChannelClassificationHead(nn.Module):

    def __init__(self, input_size, output_size, do_metric=False):
        super(ChannelClassificationHead, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc = nn.Linear(input_size, output_size)
        # self.criterion = LabelSmoothingCrossEntropy(0.05)
        self.f1 = torchmetrics.F1()
        self.do_metric = do_metric

        nn.init.xavier_uniform_(self.fc.weight)

    # @torchsnooper.snoop()
    def forward(self, feat, labels=None):
        x = self.bn(feat)
        logits = self.fc(x)

        loss = F.cross_entropy(logits, labels)
        # loss = self.criterion(logits, labels)

        if self.do_metric:
            predictions = logits.argmax(-1)
            self.f1(predictions, labels)

        return loss

    def compute(self):
        f1_score = self.f1.compute().detach().cpu()
        return f1_score

    def reset(self):
        self.f1.reset()


class EntityClassificationHead(nn.Module):

    def __init__(self, input_size, output_size, do_metric=False):
        super(EntityClassificationHead, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc = nn.Linear(input_size, output_size)
        self.f1 = torchmetrics.F1()
        self.do_metric = do_metric

        nn.init.xavier_uniform_(self.fc.weight)

    # @torchsnooper.snoop()
    def forward(self, feat, labels=None):
        logits = self.fc(self.bn(feat))

        loss = F.binary_cross_entropy_with_logits(logits, labels)

        if self.do_metric:
            predictions = logits.argmax(-1)
            self.f1(predictions, labels)

        return loss

    def compute(self):
        f1_score = self.f1.compute().detach().cpu()
        return f1_score

    def reset(self):
        self.f1.reset()


class MaskPatchRegressionHead(nn.Module):

    def __init__(self, input_size, output_size, do_metric=False):
        super(MaskPatchRegressionHead, self).__init__()

        self.bn = nn.BatchNorm1d(input_size)
        self.fc = nn.Linear(input_size, output_size)
        self.r2 = torchmetrics.R2Score()
        self.do_metric = do_metric

        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, inputs, labels):
        """

        Args:
            inputs: (batch, num_patches, feature_size)
            labels: (batch, num_patches, feature_size)

        Returns:

        """
        # Mask Patches的回归任务实现
        x = self.fc(self.bn(inputs))
        loss = F.mse_loss(x, labels)

        if self.do_metric:
            self.r2(x.reshape(-1), labels.reshape(-1))

        return loss

    def compute(self):
        r2_score = self.r2.compute().detach().cpu()
        return r2_score

    def reset(self):
        self.r2.reset()


class MaskPatchClassificationHead(nn.Module):

    def __init__(self, input_size, output_size, tau):
        super(MaskPatchClassificationHead, self).__init__()

        self.fc = nn.Linear(input_size, output_size)
        self.tau = tau
        self.epsilon = 1e-5

        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, inputs, labels):
        """

        Args:
            inputs: (batch, num_patches, feature_size)
            labels: (batch, num_patches, feature_size)

        Returns:

        """
        # Mask Patches的分类任务实现
        tau = self.tau
        batch_size, num_patches, _ = labels.size()
        broadcast_labels = (labels.unsqueeze(1) - labels.unsqueeze(2)).abs().sum(-1)  # batch, num_patches, num_patches
        idx = torch.eq(broadcast_labels, 0).float()  # find all values equal to current value
        normal_sum = idx.sum(-1).unsqueeze(-1)  # batch, num_patches, 1
        dist = idx / normal_sum  # batch, num_patches, num_patches

        x = F.normalize(self.fc(inputs), p=2, dim=-1)
        scores = torch.matmul(x, x.permute([0, 2, 1])) / tau
        probs = scores.reshape((batch_size, num_patches, num_patches)).softmax(dim=-1).clip(self.epsilon, 1 - self.epsilon)
        dist = dist.reshape((batch_size * num_patches, num_patches))
        probs = probs.reshape((batch_size * num_patches, num_patches))
        loss = F.kl_div(probs.log(), dist, reduction="batchmean")

        return loss


class MaskWordClassificationHead(nn.Module):

    def __init__(self, input_size, output_size, do_metric=False):
        super(MaskWordClassificationHead, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc = nn.Linear(input_size, output_size)
        self.acc = torchmetrics.Accuracy()
        self.do_metric = do_metric

        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, inputs, labels):
        logits = self.fc(self.bn(inputs))
        loss = F.cross_entropy(logits, labels)

        if self.do_metric:
            predictions = logits.argmax(-1)
            self.acc(predictions, labels)

        return loss

    def compute(self):
        acc_score = self.acc.compute().detach().cpu()
        return acc_score

    def reset(self):
        self.acc.reset()


if __name__ == "__main__":
    a = Dice(32)
    b = torch.zeros((10, 32))
    #b = torch.transpose(b, 1, 2)
    c = a(b)
    print(c.size())