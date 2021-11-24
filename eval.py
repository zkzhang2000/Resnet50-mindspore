# easydict模块用于以属性的方式访问字典的值
from easydict import EasyDict as edict
# os模块主要用于处理文件和目录
import os

import numpy as np
import matplotlib.pyplot as plt

import mindspore
# 导入mindspore框架数据集
import mindspore.dataset as ds
# vision.c_transforms模块是处理图像增强的高性能模块，用于数据增强图像数据改进训练模型。
from mindspore.dataset.vision import c_transforms as vision
from mindspore import context
import mindspore.nn as nn
from mindspore.train import Model
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore import Tensor
from mindspore.train.serialization import export
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.ops as ops
from resnet50 import resnet50,get_lr,read_data
import argparse


parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--checkpoint', type=str, default='./check_point/this.ckpt', help='ckpt')
args = parser.parse_args()

cfg = edict({
    'data_path': './data/new_train',  # 训练数据集，如果是zip文件需要解压
    'test_path': './data/new_test',  # 测试数据集，如果是zip文件需要解压
    'data_size': 3616,
    'HEIGHT': 224,  # 图片高度
    'WIDTH': 224,  # 图片宽度
    '_R_MEAN': 123.68,
    '_G_MEAN': 116.78,
    '_B_MEAN': 103.94,
    '_R_STD': 1,
    '_G_STD': 1,
    '_B_STD': 1,
    '_RESIZE_SIDE_MIN': 256,
    '_RESIZE_SIDE_MAX': 512,

    'batch_size': 32,
    'num_class': 108,  # 分类类别
    'epoch_size': 40,  # 训练次数
    'loss_scale_num': 1024,

    'prefix': 'resnet-ai',
    'directory': './model_resnet',
    'save_checkpoint_steps': 1000,
})
de_test = read_data(cfg.test_path,cfg,usage='test')


net = resnet50(class_num=cfg.num_class)
# 计算softmax交叉熵。
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
# 设置Adam优化器
test_step_size = de_test.get_dataset_size()
lr = Tensor(get_lr(global_step=0, total_epochs=cfg.epoch_size, steps_per_epoch=test_step_size))
opt = Momentum(net.trainable_params(), lr, momentum=0.9, weight_decay=1e-4, loss_scale=cfg.loss_scale_num)
# opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.002,
#                       0.9, 0.00004, loss_scale=1024.0)
loss_scale = FixedLossScaleManager(cfg.loss_scale_num, False)
ckpt = load_checkpoint(args.checkpoint)
load_param_into_net(net, ckpt)

model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'})
# loss_cb = LossMonitor(per_print_times=test_step_size)
# ckpt_config = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps, keep_checkpoint_max=1)
# ckpoint_cb = ModelCheckpoint(prefix=cfg.prefix, directory=cfg.directory, config=ckpt_config)

print("============== Starting Testing ==============")
# model.train(cfg.epoch_size, de_test, callbacks=[loss_cb, ckpoint_cb], dataset_sink_mode=True)


# 使用测试集评估模型，打印总体准确率
metric = model.eval(de_test)
print(metric)













