import os
from typing import Any, Callable, Dict
import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler

__all__ = ['Nusc_trainer']

class Nusc_trainer(Trainer):

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 weight_path: str = None,
                 amp_enabled: bool = False,) -> None:

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)

        self.epoch_num = 1
        self.weight_path = weight_path
        self.ignore_label = 0

    def _before_train(self) -> None:
        if self.weight_path is not None and os.path.exists(self.weight_path):
            print("load weight from", self.weight_path)
            self.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu')))
        else:
            print("train from sketch")

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:

        in_mod = {}
        in_mod['lidar'] = feed_dict['lidar'].cuda()  # [x, y, z, batch_idx] batch_idx表示这个voxel属于哪个batch
        in_mod['images'] = feed_dict['images'].permute(0, 1, 4, 2, 3).contiguous().cuda(non_blocking=True)
        in_mod['pixel_coordinates'] = [coord.cuda(non_blocking=True) for coord in feed_dict['pixel_coordinates']]
        in_mod['masks'] = [mask.cuda(non_blocking=True) for mask in feed_dict['masks']]
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)

        with amp.autocast(enabled=self.amp_enabled):
            outputs = self.model(in_mod)
            if hasattr(outputs, 'requires_grad'):
                req_grad = outputs.requires_grad
            elif isinstance(outputs, dict):
                t = np.array([getattr(v, 'requires_grad', False) for k, v in outputs.items()])
                req_grad = np.any(t)
            else:
                print("cannot figure out req_grad, default is False")
                req_grad = False
            if req_grad:
                loss_dict = self.criterion(outputs, targets)
        if req_grad:
            predict_vox = loss_dict.get('predict_vox')
            predict_pix = loss_dict.get('predict_pix')
            predict_vib = loss_dict.get('predict_vib')
            # vcd_loss = loss_dict.get('vcd_loss')
            vsd_loss = loss_dict.get('vsd_loss')
            self.summary.add_scalar('ce/vox', predict_vox.item())
            self.summary.add_scalar('ce/pix', predict_pix.item())
            self.summary.add_scalar('ce/vib', predict_vib.item())
            # self.summary.add_scalar('ib/vcd', vcd_loss.item())
            self.summary.add_scalar('ib/vsd', vsd_loss.item())
            predict_loss = (predict_vox + predict_pix + predict_vib + vsd_loss) / 4
            # predict_loss = (predict_vox + predict_pix) / 2  # uncomment this to run baseline
            loss = predict_loss
            self.summary.add_scalar('total_loss', loss.item())
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            return outputs
        else:
            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']
            _outputs_vox, _outputs_pix, _outputs_embed = [], [], []
            _targets = []
            outputs_vox = outputs.get('x_vox')
            outputs_pix = outputs.get('x_pix')
            for idx in range(invs.C[:, -1].max() + 1):
                cur_scene_pts = (in_mod['lidar'].C[:, -1] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                outputs_mapped_vox = outputs_vox[cur_scene_pts][cur_inv].argmax(1)
                outputs_mapped_pix = outputs_pix[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels.F[cur_label]
                _outputs_vox.append(outputs_mapped_vox)
                _outputs_pix.append(outputs_mapped_pix)
                _targets.append(targets_mapped)
            outputs_vox = torch.cat(_outputs_vox, 0).cpu()
            outputs_pix = torch.cat(_outputs_pix, 0).cpu()
            targets = torch.cat(_targets, 0).cpu()
            return {
                'outputs_vox': outputs_vox,
                'outputs_pix': outputs_pix,
                'targets': targets,
            }

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['scaler'] = self.scaler.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass