import torch
import torch.nn as nn
import numpy as np

# 两次升维
def header_block(in_dim, mid_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, mid_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(mid_dim),
        nn.LeakyReLU(1e-3),
        nn.Conv3d(mid_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(1e-3),
    )

# 按原文处理, 先保持维度不变, 再升维
def encoder_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(in_dim),
        nn.LeakyReLU(1e-3),
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(1e-3),
    )

# 2倍上采样
def upsample_block(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(1e-3),
    )

# 按原文处理, 先降维, 后维数不变卷积
def decoder_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(1e-3),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(1e-3),
    )

# 降维
def tail_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(1e-3),
    )

# 3D UNet, 4次降采样
# num_filter应为2的倍数
class UNet_3D(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, num_filters=4):
        super(UNet_3D, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters

        # header
        self.header = header_block(in_dim, self.num_filters//2, self.num_filters)
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # encoder
        self.encoder_1 = encoder_block(self.num_filters, self.num_filters*2)
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.encoder_2 = encoder_block(self.num_filters*2, self.num_filters*4)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.encoder_3 = encoder_block(self.num_filters*4, self.num_filters*8)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # bridge
        self.bridge = encoder_block(self.num_filters*8, self.num_filters*16)

        # decoder
        self.upsample_1 = upsample_block(self.num_filters*16, self.num_filters*16)
        self.decoder_1 = decoder_block(self.num_filters*24, self.num_filters*8)
        self.upsample_2 = upsample_block(self.num_filters*8, self.num_filters*8)
        self.decoder_2 = decoder_block(self.num_filters*12, self.num_filters*4)
        self.upsample_3 = upsample_block(self.num_filters*4, self.num_filters*4)
        self.decoder_3 = decoder_block(self.num_filters*6, self.num_filters*2)
        self.upsample_4 = upsample_block(self.num_filters*2, self.num_filters*2)
        self.decoder_4 = decoder_block(self.num_filters*3, self.num_filters)

        # output
        self.tail = tail_block(self.num_filters, self.out_dim)

    def forward(self, x):
        # x: [n, 1, 32, w*, h*]
        # pad
        w = x.shape[3]
        h = x.shape[4]
        nw = np.ceil(w / 16).astype('int32') * 16
        nh = np.ceil(h / 16).astype('int32') * 16
        inp = nn.functional.pad(x, (0, nh-h, 0, nw-w))
        # x: [n, 1, 32, 16w, 16h]

        # header
        head = self.header(inp) # -> [n, k, 32, 16w, 16h]
        pool_1 = self.pool_1(head) # -> [n, k, 16, 8w, 8h]

        # encoder
        encode_1 = self.encoder_1(pool_1) # -> [n, 2k, 16, 8w, 8h]
        pool_2 = self.pool_2(encode_1) # -> [n, 2k, 8, 4w, 4h]
        encode_2 = self.encoder_2(pool_2) # -> [n, 4k, 8, 4w, 4h]
        pool_3 = self.pool_3(encode_2) # -> [n, 4k, 4, 2w, 2h]
        encode_3 = self.encoder_3(pool_3) # -> [n, 8k, 4, 2w, 2h]
        pool_4 = self.pool_4(encode_3) # -> [n, 8k, 2, w, h]

        # bridge
        bridge = self.bridge(pool_4) # -> [n, 16k, 2, w, h]

        # decoder
        upsample_1 = self.upsample_1(bridge) # -> [n, 16k, 4, 2w, 2h]
        concat_1 = torch.cat([upsample_1, encode_3], dim=1) # -> [n, 24k, 4, 2w, 2h]
        decode_1 = self.decoder_1(concat_1) # -> [n, 8k, 4, 2w, 2h]
        upsample_2 = self.upsample_2(decode_1) # -> [n, 8k, 8, 4w, 4h]
        concat_2 = torch.cat([upsample_2, encode_2], dim=1) # -> [n, 12k, 8, 4w, 4h]
        decode_2 = self.decoder_2(concat_2) # -> [n, 4k, 8, 4w, 4h]
        upsample_3 = self.upsample_3(decode_2) # -> [n, 4k, 16, 8w, 8h]
        concat_3 = torch.cat([upsample_3, encode_1], dim=1) # -> [n, 6k, 16, 8w, 8h]
        decode_3 = self.decoder_3(concat_3) # -> [n, 2k, 16, 8w, 8h]
        upsample_4 = self.upsample_4(decode_3) # -> [n, 2k, 32, 16w, 16h]
        concat_4 = torch.cat([upsample_4, head], dim=1) # -> [n, 3k, 32, 16w, 16h]
        decode_4 = self.decoder_4(concat_4) # -> [n, k, 32, 16w, 16h]

        # output
        out = self.tail(decode_4) # -> [n, 1, 32, 16w, 16h]
        out = out[:, :, :, :x.shape[3], :x.shape[4]]
        return out

    def weight_init(self, a=1e-3):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', a))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
