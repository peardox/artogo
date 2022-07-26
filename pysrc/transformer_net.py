import torch
import time

try:
    from delphifuncts import *
    import json # Json only requited if we have delphi
except:
    have_delphi_style = False

class TransformerNet(torch.nn.Module):
    def __init__(self, showTime):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.showTime = showTime
        self.timeStamp = time.time()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1, showTime = self.showTime, timeStamp = self.timeStamp)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2, showTime = self.showTime, timeStamp = self.timeStamp)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2, showTime = self.showTime, timeStamp = self.timeStamp)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128, showTime = self.showTime, timeStamp = self.timeStamp)
        self.res2 = ResidualBlock(128, showTime = self.showTime, timeStamp = self.timeStamp)
        self.res3 = ResidualBlock(128, showTime = self.showTime, timeStamp = self.timeStamp)
        self.res4 = ResidualBlock(128, showTime = self.showTime, timeStamp = self.timeStamp)
        self.res5 = ResidualBlock(128, showTime = self.showTime, timeStamp = self.timeStamp)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, showTime = self.showTime, timeStamp = self.timeStamp, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, showTime = self.showTime, timeStamp = self.timeStamp, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1, showTime = self.showTime, timeStamp = self.timeStamp)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        if self.showTime and have_delphi_style:
            pstyle.StyleProgress(TJsonLog(event = 'styleTime', subevent = 'Controller', time = time.time() - self.timeStamp))
          
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, showTime, timeStamp):
        super(ConvLayer, self).__init__()
        self.showTime = showTime
        self.timeStamp = timeStamp
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.showTime and have_delphi_style:
            pstyle.StyleProgress(TJsonLog(event = 'styleTime', subevent = 'ConvLayer', time = time.time() - self.timeStamp))
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.showTime and have_delphi_style:
            pstyle.StyleProgress(TJsonLog(event = 'styleTime', subevent = 'ConvLayer', time = time.time() - self.timeStamp))
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, showTime, timeStamp):
        super(ResidualBlock, self).__init__()
        self.showTime = showTime
        self.timeStamp = timeStamp
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, showTime = self.showTime, timeStamp = self.timeStamp)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, showTime = self.showTime, timeStamp = self.timeStamp)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        if self.showTime and have_delphi_style:
            pstyle.StyleProgress(TJsonLog(event = 'styleTime', subevent = 'ResidualBlock', time = time.time() - self.timeStamp))
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        if self.showTime and have_delphi_style:
            pstyle.StyleProgress(TJsonLog(event = 'styleTime', subevent = 'ResidualBlock', time = time.time() - self.timeStamp))
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, showTime, timeStamp, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.showTime = showTime
        self.timeStamp = timeStamp
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            if self.showTime and have_delphi_style:
                pstyle.StyleProgress(TJsonLog(event = 'styleTime', subevent = 'UpSample', time = time.time() - self.timeStamp))
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        else:
            if self.showTime and have_delphi_style:
                pstyle.StyleProgress(TJsonLog(event = 'styleTime', subevent = 'UpSample', time = time.time() - self.timeStamp))
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        if self.showTime and have_delphi_style:
            pstyle.StyleProgress(TJsonLog(event = 'styleTime', subevent = 'UpSample', time = time.time() - self.timeStamp))
        return out
