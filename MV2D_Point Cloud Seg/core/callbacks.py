import os
import numpy as np
from typing import Any, Dict, Optional

import torch

from torchpack.environ import get_run_dir
from torchpack import distributed as dist
from torchpack.callbacks import TFEventWriter
from torchpack.callbacks.callback import Callback
from torchpack.utils import fs, io
from torchpack.utils.logging import logger
from nuscenes.eval.lidarseg.utils import ConfusionMatrix

__all__ = ['MeanIoU', 'EpochSaver']


class MeanIoU(Callback):
    def __init__(self,
                 num_classes: int,
                 ignore_label: int,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'iou') -> None:
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor

    def _before_epoch(self) -> None:
        self.total_seen = np.zeros(self.num_classes)
        self.total_correct = np.zeros(self.num_classes)
        self.total_positive = np.zeros(self.num_classes)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        if type(outputs) != np.ndarray:
            for i in range(self.num_classes):
                self.total_seen[i] += torch.sum(targets == i).item()
                self.total_correct[i] += torch.sum(
                    (targets == i) & (outputs == targets)).item()
                self.total_positive[i] += torch.sum(
                    outputs == i).item()
        else:
            for i in range(self.num_classes):
                self.total_seen[i] += np.sum(targets == i)
                self.total_correct[i] += np.sum((targets == i)
                                                & (outputs == targets))
                self.total_positive[i] += np.sum(outputs == i)

    def _after_epoch(self) -> None:
        for i in range(self.num_classes):
            self.total_seen[i] = dist.allreduce(self.total_seen[i], reduction='sum')
            self.total_correct[i] = dist.allreduce(self.total_correct[i], reduction='sum')
            self.total_positive[i] = dist.allreduce(self.total_positive[i], reduction='sum')

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                if i == self.ignore_label:
                    continue
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i] +
                                                   self.total_positive[i] -
                                                   self.total_correct[i])
                ious.append(cur_iou)

        miou = np.mean(ious)
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar(self.name, miou * 100)
            print(ious)
            print(miou)
        else:
            print(ious)
            print(miou)


class EpochSaver(Callback):
    """
    Save the checkpoint once triggered.
    """
    master_only: bool = True

    def __init__(self, *, epoch_to_save: int = 5,
                 save_dir: Optional[str] = None) -> None:
        self.epoch_to_save = epoch_to_save
        if save_dir is None:
            save_dir = os.path.join(get_run_dir(), 'checkpoints')
        self.save_dir = fs.normpath(save_dir)

    def _trigger_epoch(self) -> None:
        self._trigger()

    def _trigger(self) -> None:
        if self.trainer.epoch_num and not (self.trainer.epoch_num % self.epoch_to_save):
            save_path = os.path.join(self.save_dir,
                                     f'epoch-{self.trainer.epoch_num}.pt')
            try:
                io.save(save_path, self.trainer.state_dict())
            except OSError:
                logger.exception(
                    f'Error occurred when saving checkpoint "{save_path}".')
            else:
                logger.info(f'Checkpoint saved: "{save_path}".')