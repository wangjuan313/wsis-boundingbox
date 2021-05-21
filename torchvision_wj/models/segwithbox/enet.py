
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block_1(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )
    return model


def conv_block_3_3(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )
    return model


def conv_block_Asym(in_dim, out_dim, kernelSize):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=[kernelSize, 1], padding=tuple([2, 0])),
        nn.Conv2d(out_dim, out_dim, kernel_size=[1, kernelSize], padding=tuple([0, 2])),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )
    return model


def convBatch(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d, dilation=1):
    return nn.Sequential(
        layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
        nn.BatchNorm2d(nout),
        nn.PReLU()
    )


def upSampleConv(nin, nout, kernel_size=3, upscale=2, padding=1, bias=False):
    return nn.Sequential(
        # nn.Upsample(scale_factor=upscale),
        interpolate(mode='nearest', scale_factor=upscale),
        convBatch(nin, nout, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
        convBatch(nout, nout, kernel_size=3, stride=1, padding=1, bias=bias),
    )


class interpolate(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()

        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, cin):
        return F.interpolate(cin, mode=self.mode, scale_factor=self.scale_factor)


class BottleNeckDownSampling(nn.Module):
    def __init__(self, in_dim, projectionFactor, out_dim):
        super().__init__()

        # Main branch
        self.maxpool0 = nn.MaxPool2d(2, return_indices=True)
        # Secondary branch
        self.conv0 = nn.Conv2d(in_dim, int(in_dim / projectionFactor), kernel_size=2, stride=2)
        self.bn0 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU0 = nn.PReLU()

        self.conv1 = nn.Conv2d(int(in_dim / projectionFactor), int(in_dim / projectionFactor), kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=0.01)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):
        # Main branch
        maxpool_output, indices = self.maxpool0(input)

        # Secondary branch
        c0 = self.conv0(input)
        b0 = self.bn0(c0)
        p0 = self.PReLU0(b0)

        c1 = self.conv1(p0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)

        p2 = self.block2(p1)

        do = self.do(p2)

        # Zero padding the feature maps from the main branch
        depth_to_pad = abs(maxpool_output.shape[1] - do.shape[1])
        padding = torch.zeros(maxpool_output.shape[0], depth_to_pad, maxpool_output.shape[2],
                              maxpool_output.shape[3], device=maxpool_output.device)
        maxpool_output_pad = torch.cat((maxpool_output, padding), 1)
        output = maxpool_output_pad + do

        # _, c, _, _ = maxpool_output.shape
        # output = do
        # output[:, :c, :, :] += maxpool_output

        final_output = self.PReLU3(output)

        return final_output, indices


class BottleNeckNormal(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor, dropoutRate):
        super(BottleNeckNormal, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Main branch

        # Secondary branch
        self.block0 = conv_block_1(in_dim, int(in_dim / projectionFactor))
        self.block1 = conv_block_3_3(int(in_dim / projectionFactor), int(in_dim / projectionFactor))
        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=dropoutRate)
        self.PReLU_out = nn.PReLU()

        if in_dim > out_dim:
            self.conv_out = conv_block_1(in_dim, out_dim)

    def forward(self, input):
        # Main branch
        # Secondary branch
        b0 = self.block0(input)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        if self.in_dim > self.out_dim:
            output = self.conv_out(input) + do
        else:
            output = input + do
        output = self.PReLU_out(output)

        return output


class BottleNeckDownSamplingDilatedConv(nn.Module):
    def __init__(self, in_dim, projectionFactor, out_dim, dilation):
        super(BottleNeckDownSamplingDilatedConv, self).__init__()
        # Main branch

        # Secondary branch
        self.block0 = conv_block_1(in_dim, int(in_dim / projectionFactor))

        self.conv1 = nn.Conv2d(int(in_dim / projectionFactor), int(in_dim / projectionFactor), kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=0.01)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):
        # Secondary branch
        b0 = self.block0(input)

        c1 = self.conv1(b0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)

        b2 = self.block2(p1)

        do = self.do(b2)

        output = input + do
        output = self.PReLU3(output)

        return output


class BottleNeckNormal_Asym(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor, dropoutRate):
        super(BottleNeckNormal_Asym, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Main branch

        # Secondary branch
        self.block0 = conv_block_1(in_dim, int(in_dim / projectionFactor))
        self.block1 = conv_block_Asym(int(in_dim / projectionFactor), int(in_dim / projectionFactor), 5)
        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=dropoutRate)
        self.PReLU_out = nn.PReLU()

        if in_dim > out_dim:
            self.conv_out = conv_block_1(in_dim, out_dim)

    def forward(self, input):
        # Main branch
        # Secondary branch
        b0 = self.block0(input)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        if self.in_dim > self.out_dim:
            output = self.conv_out(input) + do
        else:
            output = input + do
        output = self.PReLU_out(output)

        return output


class BottleNeckDownSamplingDilatedConvLast(nn.Module):
    def __init__(self, in_dim, projectionFactor, out_dim, dilation):
        super(BottleNeckDownSamplingDilatedConvLast, self).__init__()
        # Main branch

        # Secondary branch
        self.block0 = conv_block_1(in_dim, int(in_dim / projectionFactor))

        self.conv1 = nn.Conv2d(int(in_dim / projectionFactor), int(in_dim / projectionFactor), kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=0.01)
        self.conv_out = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):

        # Secondary branch
        b0 = self.block0(input)

        c1 = self.conv1(b0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)

        b2 = self.block2(p1)

        do = self.do(b2)

        output = self.conv_out(input) + do
        output = self.PReLU3(output)

        return output


class BottleNeckUpSampling(nn.Module):
    def __init__(self, in_dim, projectionFactor, out_dim):
        super(BottleNeckUpSampling, self).__init__()
        # Main branch
        self.conv0 = nn.Conv2d(in_dim, int(in_dim / projectionFactor), kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU0 = nn.PReLU()

        self.conv1 = nn.Conv2d(int(in_dim / projectionFactor), int(in_dim / projectionFactor), kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=0.01)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):
        # Secondary branch
        c0 = self.conv0(input)
        b0 = self.bn0(c0)
        p0 = self.PReLU0(b0)

        c1 = self.conv1(p0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)

        p2 = self.block2(p1)

        do = self.do(p2)

        return do
    
    
class ENet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, softmax: bool, channels_in: int = 16):
        super().__init__()
        self.projecting_factor = 4
        self.n_kernels = channels_in

        # Initial
        self.conv0 = nn.Conv2d(in_dim, 15, kernel_size=3, stride=2, padding=1)
        self.maxpool0 = nn.MaxPool2d(2, return_indices=True)

        # First group
        self.bottleNeck1_0 = BottleNeckDownSampling(self.n_kernels, self.projecting_factor, self.n_kernels * 4)
        self.bottleNeck1_1 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor, 0.01)
        self.bottleNeck1_2 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor, 0.01)
        self.bottleNeck1_3 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor, 0.01)
        self.bottleNeck1_4 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor, 0.01)

        # Second group
        self.bottleNeck2_0 = BottleNeckDownSampling(self.n_kernels * 4, self.projecting_factor, self.n_kernels * 8)
        self.bottleNeck2_1 = BottleNeckNormal(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1)
        self.bottleNeck2_2 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 2)
        self.bottleNeck2_3 = BottleNeckNormal_Asym(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor,
                                                   0.1)
        self.bottleNeck2_4 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 4)
        self.bottleNeck2_5 = BottleNeckNormal(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1)
        self.bottleNeck2_6 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 8)
        self.bottleNeck2_7 = BottleNeckNormal_Asym(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor,
                                                   0.1)
        self.bottleNeck2_8 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 16)

        # Third group
        self.bottleNeck3_1 = BottleNeckNormal(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1)
        self.bottleNeck3_2 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 2)
        self.bottleNeck3_3 = BottleNeckNormal_Asym(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor,
                                                   0.1)
        self.bottleNeck3_4 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 4)
        self.bottleNeck3_5 = BottleNeckNormal(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1)
        self.bottleNeck3_6 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 8)
        self.bottleNeck3_7 = BottleNeckNormal_Asym(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor,
                                                   0.1)
        self.bottleNeck3_8 = BottleNeckDownSamplingDilatedConvLast(self.n_kernels * 8, self.projecting_factor,
                                                                   self.n_kernels * 4, 16)

        # ### Decoding path ####
        # Unpooling 1
        self.unpool_0 = nn.MaxUnpool2d(2)

        self.bottleNeck_Up_1_0 = BottleNeckUpSampling(self.n_kernels * 8, self.projecting_factor,
                                                      self.n_kernels * 4)
        self.PReLU_Up_1 = nn.PReLU()

        self.bottleNeck_Up_1_1 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor,
                                                  0.1)
        self.bottleNeck_Up_1_2 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels, self.projecting_factor, 0.1)

        # Unpooling 2
        self.unpool_1 = nn.MaxUnpool2d(2)
        self.bottleNeck_Up_2_1 = BottleNeckUpSampling(self.n_kernels * 2, self.projecting_factor, self.n_kernels)
        self.bottleNeck_Up_2_2 = BottleNeckNormal(self.n_kernels, self.n_kernels, self.projecting_factor, 0.1)
        self.PReLU_Up_2 = nn.PReLU()

        # Unpooling Last
        self.deconv3 = upSampleConv(self.n_kernels, self.n_kernels)

        self.out_025 = nn.Conv2d(self.n_kernels * 8, out_dim, kernel_size=3, stride=1, padding=1)
        self.out_05 = nn.Conv2d(self.n_kernels, out_dim, kernel_size=3, stride=1, padding=1)
        self.final = nn.Conv2d(self.n_kernels, out_dim, kernel_size=1)

        if softmax:
            self.pred_func = nn.Softmax(dim=1)
        else:
            self.pred_func = nn.Sigmoid()

        self.init_weights()

    def forward(self, input):
        conv_0 = self.conv0(input)  # This will go as res in deconv path
        maxpool_0, indices_0 = self.maxpool0(input)
        outputInitial = torch.cat((conv_0, maxpool_0), dim=1)

        # First group
        bn1_0, indices_1 = self.bottleNeck1_0(outputInitial)
        bn1_1 = self.bottleNeck1_1(bn1_0)
        bn1_2 = self.bottleNeck1_2(bn1_1)
        bn1_3 = self.bottleNeck1_3(bn1_2)
        bn1_4 = self.bottleNeck1_4(bn1_3)

        # Second group
        bn2_0, indices_2 = self.bottleNeck2_0(bn1_4)
        bn2_1 = self.bottleNeck2_1(bn2_0)
        bn2_2 = self.bottleNeck2_2(bn2_1)
        bn2_3 = self.bottleNeck2_3(bn2_2)
        bn2_4 = self.bottleNeck2_4(bn2_3)
        bn2_5 = self.bottleNeck2_5(bn2_4)
        bn2_6 = self.bottleNeck2_6(bn2_5)
        bn2_7 = self.bottleNeck2_7(bn2_6)
        bn2_8 = self.bottleNeck2_8(bn2_7)

        # Third group
        bn3_1 = self.bottleNeck3_1(bn2_8)
        bn3_2 = self.bottleNeck3_2(bn3_1)
        bn3_3 = self.bottleNeck3_3(bn3_2)
        bn3_4 = self.bottleNeck3_4(bn3_3)
        bn3_5 = self.bottleNeck3_5(bn3_4)
        bn3_6 = self.bottleNeck3_6(bn3_5)
        bn3_7 = self.bottleNeck3_7(bn3_6)
        bn3_8 = self.bottleNeck3_8(bn3_7)

        # #### Deconvolution Path ####
        #  First block #
        unpool_0 = self.unpool_0(bn3_8, indices_2)

        # bn_up_1_0 = self.bottleNeck_Up_1_0(unpool_0) # Not concatenate
        bn_up_1_0 = self.bottleNeck_Up_1_0(torch.cat((unpool_0, bn1_4), dim=1))  # concatenate

        up_block_1 = self.PReLU_Up_1(unpool_0 + bn_up_1_0)

        bn_up_1_1 = self.bottleNeck_Up_1_1(up_block_1)
        bn_up_1_2 = self.bottleNeck_Up_1_2(bn_up_1_1)

        #  Second block #
        unpool_1 = self.unpool_1(bn_up_1_2, indices_1)

        # bn_up_2_1 = self.bottleNeck_Up_2_1(unpool_1) # Not concatenate
        bn_up_2_1 = self.bottleNeck_Up_2_1(torch.cat((unpool_1, outputInitial), dim=1))  # concatenate

        bn_up_2_2 = self.bottleNeck_Up_2_2(bn_up_2_1)

        up_block_1 = self.PReLU_Up_2(unpool_1 + bn_up_2_2)

        unpool_12 = self.deconv3(up_block_1)

        output = self.final(unpool_12)
        output = self.pred_func(output)

        return [output]

    def init_weights(self, *args, **kwargs):
        # self.apply(random_weights_init)
        for m in self.modules():
            if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(m.weight.data)
            elif type(m) == nn.BatchNorm2d:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
        
# def random_weights_init(m):
#     if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
#         nn.init.xavier_normal_(m.weight.data)
#     elif type(m) == nn.BatchNorm2d:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)