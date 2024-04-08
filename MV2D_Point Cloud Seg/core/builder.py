from typing import Callable

import torch
import torch.optim
from torch import nn
import torchpack.distributed as dist
from torchpack.utils.config import configs
from torchpack.utils.typing import Dataset, Optimizer, Scheduler

__all__ = [
    'make_dataset', 'make_model', 'make_criterion', 'make_optimizer',
    'make_scheduler'
]


def make_dataset(dataset_name: str = None, **kwargs) -> dict:
    if dataset_name is None:
        dataset_name = configs.dataset.name
    if dataset_name == "semantic_nusc":
        from core.datasets import NuScenes
        dataset = NuScenes(root=configs.dataset.root,
                           voxel_size=configs.dataset.voxel_size,
                           version="v1.0-trainval",
                           verbose=True)
    elif dataset_name == "lc_semantic_nusc":
        from core.datasets.lc_semantic_nusc import LCNuScenes
        dataset = LCNuScenes(root=configs.dataset.root,
                             voxel_size=configs.dataset.voxel_size,
                             version=configs.dataset.version,
                             verbose=True,
                             image_crop_rate=configs.dataset.image_crop_rate)
    else:
        raise NotImplementedError(dataset_name)
    return dataset


def make_model(model_name=None) -> nn.Module:
    if model_name is None:
        model_name = configs.model.name
    if "cr" in configs.model:
        cr = configs.model.cr
    else:
        cr = 1.0
    if model_name == 'spvcnn':
        from core.models.nuscenes.spvcnn import SPVCNN
        model = SPVCNN(
            in_channel=configs.model.in_channel,
            num_classes=configs.data.num_classes,
            cr=cr,
            pres=configs.dataset.voxel_size,
            vres=configs.dataset.voxel_size
        )
    elif model_name == "spvcnn_swiftnet18_nusc":
        from core.models.nuscenes.spvcnn_swiftnet18 import SPVCNN_SWIFTNET18
        model = SPVCNN_SWIFTNET18(
            num_classes=configs.data.num_classes,
            cr=cr,
            pres=configs.dataset.voxel_size,
            vres=configs.dataset.voxel_size,
            imagenet_pretrain=configs.model.imagenet_pretrain,
            in_channel=configs.model.in_channel,
            proj_channel=configs.model.proj_channel
        )
    elif model_name == 'spvcnn_swiftnet_vcd':
        from core.models.nuscenes.spvcnn_swiftnet_vcd import SPVCNN_SWIFTNET_VCD
        model = SPVCNN_SWIFTNET_VCD(
            num_classes=configs.data.num_classes,
            cr=cr,
            pres=configs.dataset.voxel_size,
            vres=configs.dataset.voxel_size,
            imagenet_pretrain=configs.model.imagenet_pretrain,
            in_channel=configs.model.in_channel
        )
    else:
        raise NotImplementedError(model_name)
    return model


def make_criterion() -> Callable:
    if configs.criterion.name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=configs.criterion.ignore_index)
    elif configs.criterion.name == 'lovasz':
        from core.criterions import MixLovaszCrossEntropy
        criterion = MixLovaszCrossEntropy(ignore_index=configs.criterion.ignore_index)
    elif configs.criterion.name == 'lc_lovasz':
        from core.criterions import MixLCLovaszCrossEntropy
        criterion = MixLCLovaszCrossEntropy(ignore_index=configs.criterion.ignore_index)
    elif configs.criterion.name == 'weakly_lc_ce':
        from core.criterions import WeaklyLcLoss
        criterion = WeaklyLcLoss(ignore_index=configs.criterion.ignore_index)
    elif configs.criterion.name == 'mix_vcd_ce':
        from core.criterions import MixVCDCrossEntropy
        criterion = MixVCDCrossEntropy(ignore_index=configs.criterion.ignore_index)
    else:
        raise NotImplementedError(configs.criterion.name)
    return criterion


def make_optimizer(model: nn.Module) -> Optimizer:
    if configs.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=configs.optimizer.lr,
            momentum=configs.optimizer.momentum,
            weight_decay=configs.optimizer.weight_decay,
            nesterov=configs.optimizer.nesterov)
    elif configs.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    elif configs.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    else:
        raise NotImplementedError(configs.optimizer.name)
    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    if configs.scheduler.name == 'none':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1)
    elif configs.scheduler.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.num_epochs)
    elif configs.scheduler.name == 'cosine_warmup':
        from core.schedulers import cosine_schedule_with_warmup
        from functools import partial
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=partial(
                cosine_schedule_with_warmup,
                num_epochs=configs.num_epochs,
                batch_size=configs.batch_size,
                dataset_size=configs.data.training_size
            )
        )
    else:
        raise NotImplementedError(configs.scheduler.name)
    return scheduler
