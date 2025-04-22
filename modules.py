from torch import nn
import torch
import torch.nn.functional as F
import os  


def normalize(x, power=1):
    power_emp = torch.mean(x ** 2)
    x = (power / power_emp) ** 0.5 * x
    return power_emp, x


class DepthToSpace(torch.nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)
        return x


def awgn(snr, x, device):
    # snr(db)
    n = 1 / (10 ** (snr / 10))
    sqrt_n = n ** 0.5
    noise = torch.randn_like(x) * sqrt_n
    noise = noise.to(device)
    x_hat = x + noise
    return x_hat


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.PReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.prelu(out)
        return out


class Encoder(nn.Module):
    def __init__(self, config, prototype=None):
        super(Encoder, self).__init__()
        self.config = config
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 1, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 1, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        if config.mod_method == 'bpsk':
            self.layer4 = self.make_layer(ResidualBlock, config.channel_use, 2, stride=2)
            # 新增：升维线性层
            self.expand_dim = nn.Linear(512, config.channel_use * 4 * 4)
        else:
            self.layer4 = self.make_layer(ResidualBlock, config.channel_use * 2, 2, stride=2)
            # 新增：升维线性层
            self.expand_dim = nn.Linear(512, config.channel_use * 4 * 4 * 2)
        
        # 新增：知识库原型矩阵，初始为 None
        self.prototype = prototype  
    


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        z0 = self.conv1(x)
        z1 = self.layer1(z0)
        z2 = self.layer2(z1)
        z3 = self.layer3(z2)
        z4 = self.layer4(z3)   # z4 的 shape 为 [batch, C, H, W]
        # 如果设置了 prototype，则对 z4 进行原型融合
        if self.prototype is not None:
            # flatten z4 至 [batch, feature_dim]
            f = z4.view(z4.size(0), -1)
            # 使用线性层将原型矩阵升维到与 f 匹配的维度
            prototype_mapped = self.expand_dim(self.prototype)  # [num_classes, feature_dim]
            attn_scores = torch.matmul(f, prototype_mapped.t())  # [batch, num_classes]
            attn_weights = torch.softmax(attn_scores, dim=-1) # [batch, num_classes]
            proto_info = torch.matmul(attn_weights, prototype_mapped)  # [batch, feature_dim]
            f = f + self.config.aid_alpha * proto_info
            # 返回增强后的特征向量，此处直接返回 flattened 形式（network.py 会 reshape）
            return f
        else:
            # 如果未设置，直接返回 z4（network.py 中会 reshape为扁平向量）
            return z4


class Decoder_Recon(nn.Module):
    def __init__(self, config, prototype=None):
        super(Decoder_Recon, self).__init__()
        self.config = config
        self.prototype = prototype  # 可传入原型矩阵，用于重建增强

        if config.mod_method == 'bpsk':
            input_channel = int(config.channel_use / (4 * 4))
        else:
            input_channel = int(config.channel_use * 2 / (4 * 4))

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 256, 1, 1, 0),
            nn.PReLU())

        self.inchannel = 256

        self.layer1 = nn.Sequential(
            self.make_layer(ResidualBlock, 256, 2, stride=1),
            nn.PReLU())

        self.layer2 = nn.Sequential(
            self.make_layer(ResidualBlock, 256, 2, stride=1),
            nn.PReLU())

        self.DepthToSpace1 = DepthToSpace(4)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 128, 1, 1, 0),
            nn.PReLU())

        self.inchannel = 128

        self.layer3 = nn.Sequential(
            self.make_layer(ResidualBlock, 128, 2, stride=1),
            nn.PReLU())

        self.DepthToSpace2 = DepthToSpace(2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 3, 1, 1, 0))
        
        self.reduce_dim = nn.Linear(512, input_channel * 4 * 4 )  # 降维线性层，将原型矩阵降维到与 x 匹配的维度

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, z):

        # 如果设置了 prototype，则融合先验知识
        if self.prototype is not None:
            # Step 1: 将 prototypes 降维到与 z 匹配的维度
            prototype_mapped = self.reduce_dim(self.prototype)

            # Step 2: 计算注意力权重
            attn_scores = torch.matmul(z, prototype_mapped.t())  # [batch_size, num_classes]
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch_size, num_classes]

            # Step 3: 加权融合
            # 根据注意力权重加权 prototype_mapped，得到与 z 形状匹配的特征
            proto_info = torch.matmul(attn_weights, prototype_mapped)  # [batch_size, 128]

            # 将加权后的先验信息融合到 z 中
            z = z + self.config.aid_alpha * proto_info  # 融合超参数 aid_alpha 控制融合强度

        z0 = self.conv1(z.reshape(z.shape[0], -1, 4, 4))
        z1 = self.layer1(z0)
        z2 = self.layer2(z1)
        z3 = self.DepthToSpace1(z2)
        z4 = self.conv2(z3)
        z5 = self.layer3(z4)
        z5 = self.DepthToSpace2(z5)
        z6 = self.conv3(z5)
        return z6


class Decoder_Class(nn.Module):
    def __init__(self, half_width, layer_width, config, prototype=None):
        super(Decoder_Class, self).__init__()
        self.layer_width = layer_width
        self.Half_width = half_width
        self.prototype = prototype  # 用于分类辅助
        self.config = config
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.Half_width, self.layer_width),
            nn.PReLU(),
        )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.PReLU(),
        )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.PReLU(),
        )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.PReLU(),
        )
        self.last_fc = nn.Linear(self.layer_width * 4, 10)
        # 降维线性层，将原型矩阵降维到与 x 匹配的维度
        self.reduce_dim = nn.Linear(512, self.layer_width * 4)

    def forward(self, z):
        x1 = self.fc_spinal_layer1(z[:, 0:self.Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([z[:, self.Half_width:2 * self.Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([z[:, 0:self.Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([z[:, self.Half_width:2 * self.Half_width], x3], dim=1))
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        # 如果设置了 prototype，则利用其辅助计算分类信息
        if self.prototype is not None:
                        # Step 1: 将原型矩阵降维到与 x 匹配的维度
            prototype_reduced = self.reduce_dim(self.prototype)  # [num_classes, layer_width * 4]

            # Step 2: 计算注意力权重
            attn_scores = torch.matmul(x, prototype_reduced.t())  # [batch_size, num_classes]
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch_size, num_classes]

            # Step 3: 根据注意力权重加权原型矩阵
            proto_info = torch.matmul(attn_weights, prototype_reduced)  # [batch_size, layer_width * 4]

            # Step 4: 将加权后的原型信息与 x 融合
            x = x + self.config.aid_alpha * proto_info  # 融合超参数 aid_alpha 控制融合强度

        y_class = self.last_fc(x)
        return y_class