import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class MLP_Classifier(nn.Module):
    """
        The easy linear classifier at the end. 
    """
    def __init__(self, in_ch, out_ch) -> None:
        super(MLP_Classifier, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.fc = nn.Sequential(
            nn.Linear(in_ch, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_ch)
        )

    def forward(self, x):
        return self.fc(x)