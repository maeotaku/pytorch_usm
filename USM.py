import torch
import torch.nn as nn
from torch.nn import functional as F

from .LoG import LoG2d

class USMBase(LoG2d):

    def __init__(self, in_channels, kernel_size, fixed_coeff=False, sigma=-1, stride=1, dilation=1, cuda=False, requires_grad=True):
        #Padding must be forced so output size is = to input size
        #Thus, in_channels = out_channels
        padding = int((stride*(in_channels-1)+((kernel_size-1)*(dilation-1))+kernel_size-in_channels)/2)
        super(USMBase, self).__init__(in_channels, in_channels, kernel_size, fixed_coeff, sigma, stride, padding, dilation, cuda, requires_grad)
        self.alpha = None

    def i_weights(self):
        if self.requires_grad:
            super().init_weights()
            self.alpha.data.uniform_(0, 10)

    def assign_weight(self, alpha):
        if self.cuda:
            self.alpha = torch.cuda.FloatTensor([alpha])
        else:
            self.alpha = torch.FloatTensor([alpha])

    def forward(self, input):
        B = super().forward(input)
        U = input + self.alpha * B
        maxB = torch.max(torch.abs(B))
        maxInput = torch.max(input)
        U = U * maxInput/maxB
        return U

class USM(USMBase):
    def __init__(self, in_channels, kernel_size, fixed_coeff=False, sigma=-1, stride=1, dilation=1, cuda=False, requires_grad=True):
        super(USM, self).__init__(in_channels, kernel_size, fixed_coeff, sigma, stride, dilation, cuda, requires_grad)
        if self.requires_grad:
            if self.cuda:
                self.alpha = nn.Parameter(torch.cuda.FloatTensor(1), requires_grad=self.requires_grad)
            else:
                self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=self.requires_grad)
            self.i_weights()

class AdaptiveUSM(USMBase):
    def __init__(self, in_channels, in_side, kernel_size, fixed_coeff=False, sigma=-1, stride=1, dilation=1, cuda=False, requires_grad=True):
        super(AdaptiveUSM, self).__init__(in_channels, kernel_size, fixed_coeff, sigma, stride, dilation, cuda, requires_grad)
        if self.requires_grad:
            if self.cuda:
                self.alpha = nn.Parameter(torch.cuda.FloatTensor(in_side, in_side), requires_grad=self.requires_grad)
            else:
                self.alpha = nn.Parameter(torch.FloatTensor(in_side, in_side), requires_grad=self.requires_grad)
            self.i_weights()

'''
#CPU
x = torch.randn(2, 3, 11, 11)
l = USM(in_channels=3, kernel_size=3)
y = l(x)
print(y.size())

#CUDA
x = torch.randn(2, 3, 11, 11).cuda()
l = AdaptiveUSM(in_channels=3, in_side=11, kernel_size=3, cuda=True)
y = l(x)
print(y)
'''
