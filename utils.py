import torch
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from skimage.metrics import peak_signal_noise_ratio as comp_psnr
from skimage.metrics import structural_similarity as comp_ssim
from skimage.metrics import mean_squared_error as comp_mse
from scipy.spatial.distance import pdist, squareform


def init_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 42:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=>Saving checkpoint")
    torch.save(state, filename)


def count_percentage(code, mod, epoch, snr, channel_use, tradeoff_h):
    if mod == '4qam' or mod == 'bpsk':
        pass
    else:
        code = code.reshape(-1)
        index = [i for i in range(len(code))]
        random.shuffle(index)
        code = code[index]
        code = code.reshape(-1, 2).cpu()

        if mod == '16qam':
            I_point = torch.tensor([-3, -1, 1, 3])
            order = 16
        elif mod == '64qam':
            I_point = torch.tensor([-7, -5, -3, -1, 1, 3, 5, 7])
            order = 64

        I, Q = torch.meshgrid(I_point, I_point)
        map = torch.cat((I.unsqueeze(-1), Q.unsqueeze(-1)), dim=2).reshape(order, 2)
        per_s = []
        fig = plt.figure(dpi=300)
        ax = Axes3D(fig)
        fig.add_axes(ax)
        for i in range(order):
            temp = torch.sum(torch.abs(code - map[i, :]), dim=1)
            num = code.shape[0] - torch.count_nonzero(temp).item()
            per = num / code.shape[0]
            per_s.append(per)
        per_s = torch.tensor(per_s).cpu()
        height = np.zeros_like(per_s)
        width = depth = 0.3
        surf = ax.bar3d(I.ravel(), Q.ravel(), height, width, depth, per_s, zsort='average')
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d
        file_name = './cons_fig/' + '{}_{}_{}_{}'.format(mod, snr, channel_use, tradeoff_h)
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        fig.savefig(file_name + '/{}'.format(epoch))
        plt.close()

        # additional scatter plot
        if mod == '64qam':
            fig = plt.figure(dpi=300)
            for k in range(order):
                plt.scatter(map[k, 0], map[k, 1], s=1000 * per_s[k], color='b')
            fig.savefig(file_name + '/scatter_{}'.format(epoch))
            plt.close()


def PSNR(tensor_org, tensor_trans):
    total_psnr = 0
    origin = ((tensor_org + 1) / 2).cpu().numpy()
    trans = ((tensor_trans + 1) / 2).cpu().numpy()
    for i in range(np.size(trans, 0)):
        psnr = 0
        for j in range(np.size(trans, 1)):
            psnr_temp = comp_psnr(origin[i, j, :, :], trans[i, j, :, :])
            psnr = psnr + psnr_temp
        psnr /= 3
        total_psnr += psnr
    return total_psnr


def SSIM(tensor_org, tensor_trans):
    total_ssim = 0
    origin = tensor_org.cpu().numpy()
    trans = tensor_trans.cpu().numpy()
    for i in range(np.size(trans, 0)):
        ssim = 0
        for j in range(np.size(trans, 1)):
            ssim_temp = comp_ssim(origin[i, j, :, :], trans[i, j, :, :], data_range=1.0)
            ssim = ssim + ssim_temp
        ssim /= 3
        total_ssim += ssim

    return total_ssim


def MSE(tensor_org, tensor_trans):
    origin = ((tensor_org + 1) / 2).cpu().numpy()
    trans = ((tensor_trans + 1) / 2).cpu().numpy()
    mse = np.mean((origin - trans) ** 2)
    return mse * tensor_org.shape[0]






class PrototypeManager:
    def __init__(self, device, save_path, mismatch_level, aid_alpha):
        """
        初始化 PrototypeManager
        :param device: 设备 (CPU/GPU)
        :param save_path: 保存路径
        :param mismatch_level: 注入噪声的失配程度
        :param aid_alpha: 辅助融合参数
        """
        self.device = device
        self.save_path = save_path
        self.mismatch_level = mismatch_level
        self.aid_alpha = aid_alpha  

    # def generate_prototypes(self, encoder, dataloader, num_classes=10):
    #     """
    #     生成或加载原型矩阵
    #     :param encoder: 用于提取特征的编码器
    #     :param dataloader: 数据加载器
    #     :param num_classes: 类别数量
    #     :return: prototypes, prototypes_Kr
    #     """
    #     prototype_file = os.path.join(self.save_path, f'CIFAR_mis{self.mismatch_level:.3f}_aid{self.aid_alpha:.3f}_SKB.pt')
    #     weight_file = os.path.join(self.save_path, f'CIFAR_mis{self.mismatch_level:.3f}_aid{self.aid_alpha:.3f}_SKB.pth.tar')

    #     if os.path.exists(prototype_file):
    #         prototypes = torch.load(prototype_file).to(self.device)
    #         print(f"Loaded prototypes from {prototype_file}")
    #     else:
    #         # Load pretrained weights into the encoder if available
    #         if os.path.exists(weight_file):
    #             print(f"Loading pretrained weights from {weight_file}")
    #             state_dict = torch.load(weight_file, map_location=self.device)
    #             encoder.load_state_dict(state_dict, strict=False)

    #         # 禁用任何原型融合：确保 encoder 输出原始特征
    #         encoder.prototype = None
    #         encoder.eval()
    #         class_sum = None
    #         class_count = None
    #         with torch.no_grad():
    #             for images, labels in dataloader:
    #                 images, labels = images.to(self.device), labels.to(self.device)
    #                 feat = encoder(images)  # Ensure encoder output matches decoder input dimensions
    #                 feat = feat.view(feat.size(0), -1)
    #                 if class_sum is None:
    #                     feature_dim = feat.shape[1]
    #                     class_sum = torch.zeros(num_classes, feature_dim).to(self.device)
    #                     class_count = torch.zeros(num_classes).to(self.device)
    #                 for i in range(num_classes):
    #                     mask = (labels == i)
    #                     if mask.sum() > 0:
    #                         class_sum[i] += feat[mask].mean(dim=0)
    #                         class_count[i] += 1

    #         prototypes = class_sum / class_count.unsqueeze(1)
    #         torch.save(prototypes, prototype_file)  # prototypes shape: [num_classes, feature_dim]
    #         print(f"transmitter prototypes shape: {prototypes.shape}")
    #         print(f"Saved transmitter prototypes to {prototype_file}")

    #         # Save encoder weights
    #         torch.save(encoder.state_dict(), weight_file)
    #         print(f"Saved encoder weights to {weight_file}")
        
    #     if self.mismatch_level == 0.0:
    #         prototypes_Kr = prototypes
    #     else:
    #         # Generate Kr with injected noise
    #         prototypes_Kr = prototypes + self.mismatch_level * torch.randn_like(prototypes)
        
    #     return prototypes, prototypes_Kr

    def generate_prototypes(self, encoder, dataloader, num_classes=10,pretrained_model_path=None):
        """
        使用单独的深层网络生成原型矩阵
        """
        prototype_file = os.path.join(self.save_path, f'CIFAR_mis{self.mismatch_level:.3f}_aid{self.aid_alpha:.5f}_SKB.pt')

        if os.path.exists(prototype_file):
            prototypes = torch.load(prototype_file).to(self.device)
            print(f"Loaded prototypes from {prototype_file}")
        else:
            # 加载预训练模型参数
            if pretrained_model_path is not None:
                print(f"Loading pretrained model from {pretrained_model_path}")
                state_dict = torch.load(pretrained_model_path, map_location=self.device)
                encoder.load_state_dict(state_dict, strict=False)

            # 使用深层网络生成原型矩阵
            encoder.eval()
            class_sum = None
            class_count = None
            with torch.no_grad():
                for images, labels in dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    feat = encoder(images)  # 使用深层网络提取特征
                    feat = feat.view(feat.size(0), -1)
                    if class_sum is None:
                        feature_dim = feat.shape[1]
                        class_sum = torch.zeros(num_classes, feature_dim).to(self.device)
                        class_count = torch.zeros(num_classes).to(self.device)
                    for i in range(num_classes):
                        mask = (labels == i)
                        if mask.sum() > 0:
                            class_sum[i] += feat[mask].mean(dim=0)
                            class_count[i] += 1
    
            prototypes = class_sum / class_count.unsqueeze(1)
            torch.save(prototypes, prototype_file)  # 保存原型矩阵
            print(f"Saved prototypes to {prototype_file}")
    
        if self.mismatch_level == 0.0:
            prototypes_Kr = prototypes
        else:
            prototypes_Kr = prototypes + self.mismatch_level * torch.randn_like(prototypes)
    
        return prototypes, prototypes_Kr


    def load_prototypes(self):
            """
            加载原型矩阵（评估阶段）
            :return: prototypes, prototypes_Kr
            """
            prototype_file = os.path.join(self.save_path, f'CIFAR_mis{self.mismatch_level:.3f}_aid{self.aid_alpha:.5f}_SKB.pt')
            assert os.path.exists(prototype_file), f"Prototype file {prototype_file} does not exist!"
            prototypes = torch.load(prototype_file).to(self.device)
            print(f"Loaded prototypes from {prototype_file}")

            if self.mismatch_level == 0.0:
                prototypes_Kr = prototypes
            else:
                prototypes_Kr = prototypes + self.mismatch_level * torch.randn_like(prototypes)

            return prototypes, prototypes_Kr


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize_prototypes(prototypes, num_classes, method='tsne', save_path='./results/', file_name='prototypes.png'):
    """
    可视化原型矩阵并保存到指定路径
    :param prototypes: 原型矩阵，形状为 [num_classes, feature_dim]
    :param num_classes: 类别数量
    :param method: 降维方法 ('tsne' 或 'pca')
    :param save_path: 保存图片的路径
    :param file_name: 保存图片的文件名
    """
    prototypes = prototypes.cpu().numpy()  # 转换为 NumPy 数组

    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=min(5, num_classes - 1), random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Invalid method. Use 'tsne' or 'pca'.")

    reduced_prototypes = reducer.fit_transform(prototypes)  # 降维

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        plt.scatter(reduced_prototypes[i, 0], reduced_prototypes[i, 1], label=f'Class {i}')
    plt.title(f'Prototype Visualization ({method.upper()})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True)

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    # 保存图片
    save_file = os.path.join(save_path, file_name)
    plt.savefig(save_file, dpi=300)
    print(f"Prototype visualization saved to {save_file}")

    # 关闭绘图
    plt.close()

from torchvision.models import resnet18
import torch.nn as nn

class DeepEncoder(nn.Module):
    def __init__(self):
        super(DeepEncoder, self).__init__()
        self.resnet = resnet18(pretrained=False)  # 使用预训练的 ResNet18
        self.resnet.fc = nn.Identity()  # 移除最后的分类层，保留特征提取部分

    def forward(self, x):
        return self.resnet(x)