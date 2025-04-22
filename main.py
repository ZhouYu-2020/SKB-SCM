import torchvision
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from network import JCM
from train import train
from evaluation import EVAL
from utils import init_seeds
import os
import argparse


def mischandler(config):
    """
    初始化必要的文件夹
    """
    for path in [config.model_path, config.result_path, config.prototypes_path]:
        os.makedirs(path, exist_ok=True)


def main(config):
    # initialize random seed
    init_seeds()

    # prepare training & test data
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_data = torchvision.datasets.CIFAR10(
        root=config.dataset_path,
        train=True,
        transform=transform_train,
        download=True
    )
    test_data = torchvision.datasets.CIFAR10(
        root=config.dataset_path,
        train=False,
        transform=transform_test,
        download=True
    )

    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = JCM(config, device).to(device)

    if config.load_checkpoint:
        model_name = '/{}/'.format(config.mod_method) + \
                     'CIFAR_SNR{:.3f}_Trans{:d}_{}_mis{:.3f}_aid{:.3f}_SKB.pth.tar'.format(
                config.snr_train, config.channel_use, 
                config.mod_method,config.mismatch_level,config.aid_alpha)
        net.load_state_dict(torch.load('./checkpoints' + model_name, map_location=torch.device('cpu')))

    if config.mode == 'train':
        print("Training with the modulation scheme {}.".format(config.mod_method))
        train(config, net, train_loader, test_loader, device)

    elif config.mode == 'test':
        print("Start Testing.")
        acc, mse, psnr, ssim = EVAL(net, test_loader, device, config)
        print('acc: {:.3f}, mse: {:3f}, psnr: {:.3f}, ssmi: {:.3f}'.format(acc, mse, psnr, ssim))

    else:
        print("Wrong mode input!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--channel_use', type=int, default=128)
    """Available modulation methods:"""
    """bpsk, 4qam, 16qam, 64qam"""
    parser.add_argument('--mod_method', type=str, default='64qam')
    parser.add_argument('--load_checkpoint', type=int, default=1)

    # training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--snr_train', type=float, default=18)
    parser.add_argument('--snr_test', type=float, default=18)
    """The tradeoff hyperparameter lambda between two tasks"""
    parser.add_argument('--tradeoff_lambda', type=float, default=200)
    
    # The mismatch between K_t and K_r
    parser.add_argument('--mismatch_level', type=float, default=0.5)
    """ 背景知识的融合超参数 """
    parser.add_argument('--aid_alpha', type=float, default=0.5)

    # misc
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--dataset_path', type=str, default='./dataset')
    parser.add_argument('--prototypes_path', type=str, default='./prototypes')

    config = parser.parse_args()

    config.mode='test'
    config.mod_method='bpsk'
    config.mismatch_level=0.0
    config.aid_alpha=0.001
    config.snr_train=18
    config.snr_test=18
    config.load_checkpoint=1

    mischandler(config)
    main(config)

