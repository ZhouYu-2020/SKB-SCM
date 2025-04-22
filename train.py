import os
import torch
from torch import optim, nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from evaluation import EVAL
from utils import save_checkpoint, PSNR, PrototypeManager
from modules import Encoder

from utils import visualize_prototypes, DeepEncoder




def train(config, net, train_iter, test_iter, device):
    learning_rate = config.lr
    epochs = config.train_iters

    # lr for prob_conv needs separate setting
    ignored_params = list(map(id, net.prob_convs.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    optimizer = optim.Adam([
        {'params': base_params},
        {'params': net.prob_convs.parameters(), 'lr': learning_rate/2}], learning_rate)

    loss_f1 = nn.CrossEntropyLoss()
    loss_f2 = nn.MSELoss()
    results = {'epoch': [], 'acc': [], 'mse': [], 'psnr': [], 'ssim': [], 'loss': []}
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.train_iters+1, T_mult=1, eta_min=1e-6, last_epoch=-1)

    # 初始化 PrototypeManager
    prototype_manager = PrototypeManager(
        device=device,
        save_path=config.prototypes_path,
        mismatch_level=config.mismatch_level,
        aid_alpha=config.aid_alpha)

    # 生成或加载原型矩阵
    encoder_for_proto = DeepEncoder().to(device)
    prototypes, prototypes_Kr = prototype_manager.generate_prototypes(
        encoder=encoder_for_proto,
        dataloader=train_iter,
        num_classes=10
    )
    # 将生成的原型矩阵赋值给模型中 Encoder 和 Decoder
    with torch.no_grad():
        net.encoder.prototype = prototypes           # 发端使用原型矩阵
        net.decoder_recon.prototype = prototypes_Kr   # 解码器（重建）使用注入失配的原型
        net.decoder_class.prototype = prototypes_Kr   # 解码器（分类）同样使用
    print("Prototype matrices have been set in Encoder and Decoders.")


    # 可视化原型矩阵
    visualize_prototypes(prototypes, num_classes=10, method='tsne', save_path=config.result_path, file_name='prototypes_tsne.png')
    print(f"Prototype matrix mean: {prototypes.mean().item():.4f}, std: {prototypes.std().item():.4f}")
    print(f"Prototype matrix min: {prototypes.min().item():.4f}, max: {prototypes.max().item():.4f}")


    best_acc = 0
    for epoch in range(epochs):
        net.train()
        epoch_loss = []
        acc_total_train = 0
        psnr_total_train = 0
        for i, (X, Y) in enumerate(tqdm(train_iter)):
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            code, _, _, y_class, y_recon = net(X)

            loss_1 = loss_f1(y_class, Y)
            loss_2 = loss_f2(y_recon, X)

            loss = loss_1 + config.tradeoff_lambda * loss_2

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.cpu().item())

            # acc & psnr of the train set
            acc = (y_class.data.max(1)[1] == Y.data).float().sum()
            acc_total_train += acc
            psnr = PSNR(X, y_recon.detach())
            psnr_total_train += psnr

        scheduler.step()

        loss = sum(epoch_loss) / len(epoch_loss)
        acc_train = acc_total_train / 50000
        psnr_train = psnr_total_train / 50000

        acc, mse, psnr, ssim = EVAL(net, test_iter, device, config, epoch)
        print('epoch: {:d}, loss: {:.6f}, acc: {:.3f}, mse: {:.6f}, psnr: {:.3f}, ssim: {:.3f}, lr: {:.6f}'.format
              (epoch, loss, acc, mse, psnr, ssim, optimizer.state_dict()['param_groups'][0]['lr']))
        print('train acc: {:.3f}'.format(acc_train))
        print('train psnr: {:.3f}'.format(psnr_train))

        acc_num = acc.detach().cpu().numpy()
        results['epoch'].append(epoch)
        results['loss'].append(loss)
        results['acc'].append(acc_num)
        results['mse'].append(mse)
        results['psnr'].append(psnr)
        results['ssim'].append(ssim)

        if (epochs - epoch) <= 10 and acc_num > best_acc:
            file_name = config.model_path + '/{}/'.format(config.mod_method)
            if not os.path.exists(file_name):
                os.makedirs(file_name)
            model_name = 'CIFAR_SNR{:.3f}_Trans{:d}_{}_mis{:.3f}_aid{:.3f}_SKB.pth.tar'.format(
                config.snr_train, config.channel_use, 
                config.mod_method,config.mismatch_level,config.aid_alpha)
            save_checkpoint(net.state_dict(), file_name + model_name)
            best_acc = acc_num

    # in the end save all the results
    data = pd.DataFrame(results)
    file_name = config.result_path + '/{}/'.format(config.mod_method)
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    result_name = 'CIFAR_SNR{:.3f}_Trans{:d}_{}.csv'.format(
            config.snr_train, config.channel_use, config.mod_method)
    data.to_csv(file_name + result_name, index=False, header=False)


