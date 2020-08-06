
import torch
import torch.nn as nn
class UNet(nn.Module):
    def __init__(self, in_channels,channels, n_classes):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.channels=channels
        self.n_classes = n_classes
        self.inc = DoubleConv(in_channels, channels)
        self.down1 = Down(channels)
        self.down2 = Down(channels)
        self.down3 = Down(channels)
        self.down4 = Down(channels)
        self.up1 = Up(channels)
        self.up2 = Up(channels)
        self.up3 = Up(channels)
        self.up4 = Up(channels)
        self.outc = OutConv2d(channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        feature = self.up4(x, x1)
        logits = self.outc(feature)
        return logits,feature
class UNet_3Plus(nn.Module):

    def __init__(self, in_channels,channels=64,n_classes=2):
        super(UNet_3Plus, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes=n_classes

        ## -------------Encoder--------------
        self.conv1 = DoubleConv(self.in_channels, self.channels)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = DoubleConv(self.channels, self.channels)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = DoubleConv(self.channels, self.channels)

        ## -------------Decoder--------------
        self.CatChannels = self.channels
        self.CatBlocks = 3
        self.UpChannels = self.CatChannels * self.CatBlocks

        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(self.channels, self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(self.channels, self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        self.h3_Cat_hd3_conv = nn.Conv2d(self.channels, self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)


        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(self.channels, self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        self.h2_Cat_hd2_conv = nn.Conv2d(self.channels, self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.relu2d_1 = nn.ReLU(inplace=True)


        self.h1_Cat_hd1_conv = nn.Conv2d(self.channels, self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)  # 16
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.outconv1 = nn.Conv2d(self.CatChannels, n_classes, 3, padding=1)


    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->304*304

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->152*152

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->76*76


        ## -------------Decoder-------------

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1)))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2)))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_conv(h3))
        hd3 = self.relu3d_1(self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3), 1)))

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1)))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_conv(h2))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))
        hd2 = self.relu2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2), 1)))

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_conv(h1))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3)))
        x = self.relu1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1), 1)))

        d1 = self.outconv1(x)  # d1->320*320*n_classes
        return d1,x
class IPN(nn.Module):
    def __init__(self, in_channels,channels, n_classes):
        super(IPN, self).__init__()
        self.in_channels = in_channels
        self.channels=channels
        self.n_classes = n_classes
        self.input = InConv3d(in_channels,channels)
        self.PLM1 = PLM(5,channels)
        self.PLM2 = PLM(4,channels)
        self.PLM3 = PLM(4,channels)
        self.PLM4 = PLM(2,channels)
        self.output = OutConv3d(channels,n_classes)

    def forward(self, x):
        x = self.input(x)
        x = self.PLM1(x)
        x = self.PLM2(x)
        x = self.PLM3(x)
        feature = self.PLM4(x)
        feature = torch.squeeze(feature, 2)
        logits = self.output(feature)
        return logits,feature
class IPN_V2(nn.Module):
    def __init__(self, in_channels,channels,plane_perceptron_channels,n_classes,block_size,plane_perceptron):
        super(IPN_V2, self).__init__()
        self.in_channels = in_channels
        self.channels=channels
        self.n_classes = n_classes
        self.planar_perceptron_channels=plane_perceptron_channels
        self.input3d = InConv3d(in_channels,channels)
        self.PLM1 = PLM(8,channels)
        self.PLM2 = PLM(5,channels)
        self.PLM3 = PLM(4,channels)
        self.input2d = skip(channels,block_size[0])
        if plane_perceptron=='UNet_3Plus':
            self.pp= UNet_3Plus(channels,plane_perceptron_channels, n_classes)
        if plane_perceptron=='UNet':
            self.pp= UNet(channels,plane_perceptron_channels, n_classes)

    def forward(self, x0):
        x0 = self.input3d(x0)
        x = self.PLM1(x0)
        x = self.PLM2(x)
        x = self.PLM3(x)
        x1 = self.input2d(x0,x)
        output,feature = self.pp(x1)
        logits=torch.unsqueeze(output,2)
        return logits,feature
class InConv3d(nn.Module):

    def __init__(self, in_channels, channels):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv3d(in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.in_conv(x)
class skip(nn.Module):

    def __init__(self, channels,cube_height):
        super().__init__()
        self.double2dconv = DoubleConv(channels*2,channels)
        #self.compress_pool =nn.MaxPool3d(kernel_size=[cube_height, 1, 1])
        self.skip_conv  = nn.Conv3d(channels, channels, kernel_size=[cube_height, 1, 1], stride=[cube_height, 1, 1])


    def forward(self, x0, x):
        x1 = self.skip_conv(x0)
        x = torch.cat([x1, x], dim=1)
        x = torch.squeeze(x, 2)
        return self.double2dconv(x)
class PLM(nn.Module):

    def __init__(self, poolingsize, channels):
        super().__init__()
        self.plm = nn.Sequential(
            nn.MaxPool3d(kernel_size=[poolingsize, 1, 1]),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.plm(x)
class OutConv2d(nn.Module):
    def __init__(self, channels, n_class):
        super(OutConv2d, self).__init__()
        self.conv = nn.Conv2d(channels, n_class, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class OutConv3d(nn.Module):

    def __init__(self, channels,n_class):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv3d(channels, n_class, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.out_conv(x)
class DoubleConv(nn.Module):
    """(convolution=> ReLU) * 2"""

    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,  kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class Double3DConv(nn.Module):
    """(convolution=> ReLU) * 2"""

    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.double_3dconv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_3dconv(x)
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(channels,channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, channels,  bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(channels // 2, channels // 2, kernel_size=2, stride=2)  #cyr6e# channels ?

        self.conv = DoubleConv(channels*2,channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
