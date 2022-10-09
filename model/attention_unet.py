import torch
import torch.nn as nn


class unet_with_attention(nn.Module):
    def __init__(self, net_in_channels=1, net_out_channels=1):
        super(unet_with_attention, self).__init__()

        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.max_channel_depth = 128
        self.unet_input_channels = (net_out_channels + self.max_channel_depth // 8) // 2

        self.Conv_block_1_down = conv_block(input_channels=net_in_channels, output_channels=self.unet_input_channels)
        self.Conv_block_2_down = conv_block(input_channels=self.unet_input_channels, output_channels=self.max_channel_depth // 8)
        self.Conv_block_3_down = conv_block(input_channels=self.max_channel_depth // 8, output_channels=self.max_channel_depth // 4)
        self.Conv_block_4_down = conv_block(input_channels=self.max_channel_depth // 4, output_channels=self.max_channel_depth // 2)
        self.Conv_block_5_down = conv_block(input_channels=self.max_channel_depth // 2, output_channels=self.max_channel_depth)

        self.Up_sample_block_2 = upsample_block(input_channels=self.max_channel_depth // 8, output_channels=self.unet_input_channels)
        self.Up_sample_block_3 = upsample_block(input_channels=self.max_channel_depth // 4, output_channels=self.max_channel_depth // 8)
        self.Up_sample_block_4 = upsample_block(input_channels=self.max_channel_depth // 2, output_channels=self.max_channel_depth // 4)
        self.Up_sample_block_5 = upsample_block(input_channels=self.max_channel_depth, output_channels=self.max_channel_depth // 2)

        self.Attention_block_2 = attention_Block(F_g=self.unet_input_channels, F_l=self.unet_input_channels, F_int=self.unet_input_channels // 2)
        self.Attention_block_3 = attention_Block(F_g=self.max_channel_depth // 8, F_l=self.max_channel_depth // 8, F_int=self.max_channel_depth // 16)
        self.Attention_block_4 = attention_Block(F_g=self.max_channel_depth // 4, F_l=self.max_channel_depth // 4, F_int=self.max_channel_depth // 8)
        self.Attention_block_5 = attention_Block(F_g=self.max_channel_depth // 2, F_l=self.max_channel_depth // 2, F_int=self.max_channel_depth // 4)

        self.Conv_block_2_up = conv_block(input_channels=2 * self.unet_input_channels, output_channels=self.unet_input_channels)
        self.Conv_block_3_up = conv_block(input_channels=self.max_channel_depth // 4, output_channels=self.max_channel_depth // 8)
        self.Conv_block_4_up = conv_block(input_channels=self.max_channel_depth // 2, output_channels=self.max_channel_depth // 4)
        self.Conv_block_5_up = conv_block(input_channels=self.max_channel_depth, output_channels=self.max_channel_depth // 2)

        self.Conv_output = nn.Conv1d(self.unet_input_channels, net_out_channels, kernel_size=(1,), stride=(1,), padding=0)

    def forward(self, net_in):
        # encoding
        unet_in = self.Conv_block_1_down(net_in)

        in2 = self.Maxpool(unet_in)
        in2 = self.Conv_block_2_down(in2)

        in3 = self.Maxpool(in2)
        in3 = self.Conv_block_3_down(in3)

        in4 = self.Maxpool(in3)
        in4 = self.Conv_block_4_down(in4)

        in5 = self.Maxpool(in4)
        in5 = self.Conv_block_5_down(in5)

        # decoding + skip connection
        out5 = self.Up_sample_block_5(in5)
        atten5_out = self.Attention_block_5(g=out5, x=in4)
        attended5 = torch.cat((atten5_out, out5), dim=1)
        out5 = self.Conv_block_5_up(attended5)

        out4 = self.Up_sample_block_4(out5)
        atten4_out = self.Attention_block_4(g=out4, x=in3)
        attended4 = torch.cat((atten4_out, out4), dim=1)
        out4 = self.Conv_block_4_up(attended4)

        out3 = self.Up_sample_block_3(out4)
        atten3_out = self.Attention_block_3(g=out3, x=in2)
        attended3 = torch.cat((atten3_out, out3), dim=1)
        out3 = self.Conv_block_3_up(attended3)

        out2 = self.Up_sample_block_2(out3)
        atten2_out = self.Attention_block_2(g=out2, x=unet_in)
        attended2 = torch.cat((atten2_out, out2), dim=1)
        unet_out = self.Conv_block_2_up(attended2)

        net_out = self.Conv_output(unet_out)

        return net_out


class conv_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(conv_block, self).__init__()
        self.kernel = 11
        self.pad = (self.kernel - 1) // 2
        self.conv_block_net = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=(self.kernel,), stride=(1,), padding=self.pad, bias=True),
            nn.BatchNorm1d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(output_channels, output_channels, kernel_size=(self.kernel,), stride=(1,), padding=self.pad, bias=True),
            nn.BatchNorm1d(output_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, net_in):
        net_out = self.conv_block_net(net_in)
        return net_out


class upsample_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(upsample_block, self).__init__()
        self.up_sample_block_net = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(input_channels, output_channels, kernel_size=(11,), stride=(1,), padding=5, bias=True),
            nn.BatchNorm1d(output_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, net_in):
        net_out = self.up_sample_block_net(net_in)
        return net_out


# based on: Attention U-Net:Learning Where to Look for the Pancreas
# https://arxiv.org/pdf/1804.03999.pdf

class attention_Block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(attention_Block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=(1,), stride=(1,), padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=(1,), stride=(1,), padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=(1,), stride=(1,), padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
