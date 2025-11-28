"""Experiment class for multi-dataset univariate long-term forecasting."""

from pathlib import Path
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.multi_dataset import multi_dataset_provider, load_multi_dataset_specs
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

warnings.filterwarnings('ignore')


class Exp_Multi_Dataset_Forecast(Exp_Basic):
    """Multi-dataset univariate long-term forecasting experiment.

    Uses --features S for all datasets, training on concatenated training sets,
    validating on concatenated validation sets, and testing on each dataset
    separately to report per-dataset metrics.
    """

    def __init__(self, args):
        super().__init__(args)
        self.spec_path = Path(args.multi_dataset_spec)
        if not self.spec_path.exists():
            raise FileNotFoundError(f"Multi-dataset spec not found: {self.spec_path}")
        self.dataset_names = None

    def _unpack_model_output(self, model_output):
        if isinstance(model_output, (list, tuple)):
            pred = model_output[0]
            aux = model_output[1] if len(model_output) > 1 else None
            return pred, aux
        return model_output, None

    def _compute_moe_aux_loss(self, router_probs):
        coeff = getattr(self.args, 'moe_aux_loss_coeff', 0.0)
        if coeff <= 0 or router_probs is None:
            return None
        expert_mean = router_probs.mean(dim=0)
        uniform = torch.full_like(expert_mean, 1.0 / expert_mean.numel())
        aux_loss = (expert_mean - uniform).pow(2).sum() * expert_mean.numel()
        return coeff * aux_loss

    def _get_base_model(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def _build_model(self):
        # Override enc_in, dec_in, c_out for univariate
        self.args.enc_in = 1
        self.args.dec_in = 1
        self.args.c_out = 1
        self.args.features = 'S'

        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        dataset, loader, names = multi_dataset_provider(
            spec_path=self.spec_path,
            flag=flag,
            seq_len=self.args.seq_len,
            label_len=self.args.label_len,
            pred_len=self.args.pred_len,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            args_template=self.args,
        )
        if self.dataset_names is None:
            self.dataset_names = names
        return dataset, loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs, _ = self._unpack_model_output(outputs)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                loss = criterion(outputs.detach(), batch_y.detach())
                total_loss.append(loss.item())

        self.model.train()
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # LoMoE phase control
        base_model = self._get_base_model()
        warmup_epochs = max(0, getattr(self.args, 'lomoe_warmup_epochs', 0))
        supports_lomoe_control = all(
            hasattr(base_model, attr)
            for attr in ['set_single_expert_mode', 'replicate_primary_expert', 'set_cluster_router_enabled']
        )
        supports_lora_freeze = hasattr(base_model, 'freeze_backbone_for_lora')
        freeze_backbone_after_warmup = bool(getattr(self.args, 'lomoe_freeze_backbone_after_warmup', False))
        warmup_requested = warmup_epochs > 0
        warmup_active = supports_lomoe_control and warmup_requested
        phase2_lr_scale = float(getattr(self.args, 'lomoe_phase2_lr_scale', 1.0))
        phase2_lr_pending = warmup_active and abs(phase2_lr_scale - 1.0) > 1e-9
        phase2_lr_applied = False

        if warmup_active:
            base_model.set_single_expert_mode(0)
            base_model.set_cluster_router_enabled(False)
            print("[LoMoE] Warmup enabled: training with expert 0 only until patience triggers.")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs, router_probs = self._unpack_model_output(outputs)
                        outputs = outputs[:, -self.args.pred_len:, :]
                        batch_y = batch_y[:, -self.args.pred_len:, :]
                        task_loss = criterion(outputs, batch_y)
                        aux_loss = self._compute_moe_aux_loss(router_probs)
                        loss = task_loss + (aux_loss if aux_loss is not None else 0)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs, router_probs = self._unpack_model_output(outputs)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :]
                    task_loss = criterion(outputs, batch_y)
                    aux_loss = self._compute_moe_aux_loss(router_probs)
                    loss = task_loss + (aux_loss if aux_loss is not None else 0)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, path)

            # LoMoE phase transition
            if warmup_active and early_stopping.early_stop:
                checkpoint_path = os.path.join(path, 'checkpoint.pth')
                if os.path.exists(checkpoint_path):
                    self.model.load_state_dict(torch.load(checkpoint_path))
                    print("[LoMoE] Loaded best checkpoint from warmup before entering phase two.")
                base_model.replicate_primary_expert(0)
                base_model.set_single_expert_mode(None)
                base_model.set_cluster_router_enabled(True)
                if freeze_backbone_after_warmup and supports_lora_freeze:
                    base_model.freeze_backbone_for_lora()
                    print("[LoMoE] Backbone frozen: continuing training with LoRA experts only.")
                if phase2_lr_pending and not phase2_lr_applied:
                    current_lr = model_optim.param_groups[0]['lr']
                    new_lr = current_lr * phase2_lr_scale
                    for param_group in model_optim.param_groups:
                        param_group['lr'] = new_lr
                    self.args.learning_rate = new_lr
                    phase2_lr_applied = True
                    print(f"[LoMoE] Phase 2 LR scaling applied: {current_lr:.6e} -> {new_lr:.6e}.")
                warmup_active = False
                phase2_lr_pending = False
                early_stopping.reset()
                print("[LoMoE] Warmup ended: replicated expert 0 weights, re-enabled router, and reset early stopping.")

            if not warmup_active and early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        """Test on each dataset separately and report per-dataset metrics."""
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        specs = load_multi_dataset_specs(self.spec_path)
        all_results = {}

        for spec in specs:
            print(f"\n=== Testing on {spec.name} ===")
            preds, trues = self._test_single_dataset(spec, setting)

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print(f'{spec.name}: mse={mse:.6f}, mae={mae:.6f}')
            all_results[spec.name] = {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape, 'mspe': mspe}

        # Aggregate results
        avg_mse = np.mean([r['mse'] for r in all_results.values()])
        avg_mae = np.mean([r['mae'] for r in all_results.values()])
        print(f"\n=== Average across datasets: mse={avg_mse:.6f}, mae={avg_mae:.6f} ===")

        # Save results
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(os.path.join(folder_path, 'multi_dataset_results.txt'), 'w') as f:
            for name, metrics in all_results.items():
                f.write(f"{name}: mse={metrics['mse']:.6f}, mae={metrics['mae']:.6f}\n")
            f.write(f"\nAverage: mse={avg_mse:.6f}, mae={avg_mae:.6f}\n")

        # Also append to global results
        with open("result_multi_dataset_forecast.txt", 'a') as f:
            f.write(setting + "\n")
            for name, metrics in all_results.items():
                f.write(f"  {name}: mse={metrics['mse']:.6f}, mae={metrics['mae']:.6f}\n")
            f.write(f"  Average: mse={avg_mse:.6f}, mae={avg_mae:.6f}\n\n")

        return all_results

    def _test_single_dataset(self, spec, setting):
        """Test on a single dataset."""
        from data_provider.data_factory import data_dict

        Data = data_dict[spec.data]
        timeenc = 0 if getattr(self.args, 'embed', 'timeF') != 'timeF' else 1

        ds_args = type('Args', (), {'augmentation_ratio': 0})()
        test_data = Data(
            args=ds_args,
            root_path=spec.root_path,
            data_path=spec.data_path,
            flag='test',
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features='S',
            target=spec.target,
            timeenc=timeenc,
            freq=spec.freq,
            seasonal_patterns='monthly',
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
        )

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs, _ = self._unpack_model_output(outputs)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        return preds, trues
