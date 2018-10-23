import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import conv2d

def LoG_np(k, sigma):
    ax = np.round(np.linspace(-np.floor(k/2),np.floor(k/2),k));
    x, y = np.meshgrid(ax, ax)
    x2 = np.power(x,2)
    y2 = np.power(y,2)
    s2 = np.power(sigma,2)
    s4 = np.power(sigma,4)
    hg = np.exp(-(x2 + y2)/(2.*s2))
    kernel_t = hg*(x2 + y2-2*s2)/(s4*np.sum(hg))
    kernel = kernel_t - np.sum(kernel_t)/np.power(k,2);
    return kernel

def log2d(k, sigma, cuda=False):
    if cuda:
        ax = torch.round(torch.linspace(-math.floor(k/2), math.floor(k/2), k), out=torch.FloatTensor());
        ax = ax.cuda()
    else:
        ax = torch.round(torch.linspace(-math.floor(k/2), math.floor(k/2), k), out=torch.FloatTensor());
    y = ax.view(-1, 1).repeat(1, ax.size(0))
    x = ax.view(1, -1).repeat(ax.size(0), 1)
    x2 = torch.pow(x, 2)
    y2 = torch.pow(y, 2)
    s2 = torch.pow(sigma, 2)
    s4 = torch.pow(sigma, 4)
    hg = (-(x2 + y2)/(2.*s2)).exp()
    kernel_t = hg*(1.0 - (x2 + y2 / 2*s2)) * (1.0 / s4 * hg.sum())
    if cuda:
        kernel = kernel_t - kernel_t.sum() / torch.pow(torch.FloatTensor([k]).cuda(),2)
    else:
        kernel = kernel_t - kernel_t.sum() / torch.pow(torch.FloatTensor([k]),2)
    return kernel
    '''
    hg = (-(x2 + y2)/(2.*s2)).exp()
    kernel_t = hg*(x2 + y2-2*s2) /(s4*hg.sum())
    if cuda:
        kernel = kernel_t - kernel_t.sum() / torch.pow(torch.FloatTensor([k]).cuda(),2)
    else:
        kernel = kernel_t - kernel_t.sum() / torch.pow(torch.FloatTensor([k]),2)
    #kernel = torch.FloatTensor([[0.0, -1.0, 0.0],[-1.0, 4.0, -1.0],[0.0, -1.0, 0.0]])
    return kernel
    '''


class LoG2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, fixed_coeff=False, sigma=-1, stride=1, padding=0, dilation=1, cuda=False, requires_grad=True):
        super(LoG2d, self).__init__()
        self.fixed_coeff = fixed_coeff
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.cuda = cuda
        self.requires_grad = requires_grad
        if not self.fixed_coeff:
            if self.cuda:
                self.sigma = nn.Parameter(torch.cuda.FloatTensor(1), requires_grad=self.requires_grad)
            else:
                self.sigma = nn.Parameter(torch.FloatTensor(1), requires_grad=self.requires_grad)
        else:
            if self.cuda:
                self.sigma = torch.cuda.FloatTensor([sigma])
            else:
                self.sigma = torch.FloatTensor([sigma])
            self.kernel = log2d(self.kernel_size, self.sigma, self.cuda)
            self.kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
            self.kernel = self.kernel.repeat(self.out_channels, 1, 1, 1)
        self.init_weights()

    def init_weights(self):
        if not self.fixed_coeff:
            self.sigma.data.uniform_(0.0001, 0.9999)

    def forward(self, input):
        if not self.fixed_coeff:
            self.kernel = log2d(self.kernel_size, self.sigma, self.cuda)
            self.kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
            self.kernel = self.kernel.repeat(self.out_channels, 1, 1, 1)
        kernel = self.kernel
        #kernel size is (out_channels, in_channels, h, w)
        res = conv2d(input, kernel, padding=self.padding, groups=self.out_channels)#, stride=self.stride, padding=self.padding, dilation=self.dilation)
        return res

'''
#CPU
print(LoG_np(5, 1.4))
sigma = torch.FloatTensor([1.4])
print(LoG_2d(5, sigma))
x = torch.randn(2, 3, 11, 11) #batch=1, channels=1, res=7x7
l = LoG2d(in_channels=3, out_channels=3, kernel_size=3)
y = l(x)
'''

'''
#CUDA
#print(LoG_np(3, 1.4))
sigma = torch.cuda.FloatTensor([1.4])
print(LoG_2d(3, sigma, cuda=True))
x = torch.randn(2, 3, 11, 11).cuda() #batch=1, channels=1, res=7x7
l = LoG2d(in_channels=3, out_channels=3, kernel_size=3, cuda=True)
y = l(x)
print(y.size())
'''
