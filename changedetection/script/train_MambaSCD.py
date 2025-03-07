import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname((os.path.dirname(__file__))))))

import argparse
import os
import time

import numpy as np

from RemoteSensing.changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from RemoteSensing.changedetection.datasets.make_data_loader import SemanticChangeDetectionDatset, make_data_loader
from RemoteSensing.changedetection.utils_func.metrics import Evaluator
from RemoteSensing.changedetection.models.STMambaSCD import STMambaSCD
import RemoteSensing.changedetection.utils_func.lovasz_loss as L
from torch.optim.lr_scheduler import StepLR
from RemoteSensing.changedetection.utils_func.mcd_utils import accuracy, SCDD_eval_all, AverageMeter

from RemoteSensing.changedetection.utils_func.loss import contrastive_loss, ce2_dice1, ce2_dice1_multiclass

from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)

        self.deep_model = STMambaSCD(
            output_cd = 2, 
            output_clf = 7,
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            ) 
        self.deep_model = self.deep_model.cuda()
        self.model_save_path = os.path.join(args.model_param_path, "SeK_Highest")
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

        self.scheduler = StepLR(self.optim, step_size=10000, gamma=0.5)

        self.writer = SummaryWriter(log_dir=os.path.join(self.model_save_path, 'logs'))

        self.global_iter = 0

    def validate_diffusion(self):
        """Validation function for the diffusion branch."""
        self.deep_model.eval()
        dataset = SemanticChangeDetectionDatset(
            self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test'
        )
        self.val_data_loader = DataLoader(dataset, batch_size=8, num_workers=4, drop_last=False)
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for data in self.val_data_loader:
                pre_change_imgs, post_change_imgs, *_ = data
                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                x = torch.cat([pre_change_imgs, post_change_imgs], dim=1)  # [B,6,H,W]
                batch_size = x.shape[0]
                timesteps = torch.randint(0, 1000, (batch_size,), device=x.device).long()
                noise = torch.randn_like(x)
                x_noisy = self.deep_model.diffusion.q_sample(x, timesteps, noise=noise)
                timesteps = timesteps.float()
                predicted_noise, _ = self.deep_model.diffusion.model(x_noisy, timesteps)
                loss = F.mse_loss(predicted_noise, noise)
                total_loss += loss.item()
                num_batches += 1
                # Optionally limit validation batches.
                if num_batches >= 10:
                    break
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.deep_model.train()
        return avg_loss

    def validate_detection(self):
        print('---------starting evaluation-----------')
        dataset = SemanticChangeDetectionDatset(
            self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test'
        )
        val_data_loader = DataLoader(dataset, batch_size=10, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()
        acc_meter = AverageMeter()
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels_cd, labels_clf_t1, labels_clf_t2, _ = data
                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_cd = labels_cd.cuda().long()
                labels_clf_t1 = labels_clf_t1.cuda().long()
                labels_clf_t2 = labels_clf_t2.cuda().long()
                output_1, output_semantic_t1, output_semantic_t2, _ = self.deep_model(pre_change_imgs, post_change_imgs)
                labels_cd = labels_cd.cpu().numpy()
                labels_A = labels_clf_t1.cpu().numpy()
                labels_B = labels_clf_t2.cpu().numpy()
                change_mask = torch.argmax(output_1, axis=1).cpu().numpy()
                preds_A = torch.argmax(output_semantic_t1, dim=1).cpu().numpy()
                preds_B = torch.argmax(output_semantic_t2, dim=1).cpu().numpy()
                preds_A[change_mask == 0] = 0
                preds_B[change_mask == 0] = 0
                if itera % 100 == 0:
                    print(f'Validation iteration {itera}')
                for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                    acc_A, _ = accuracy(pred_A, label_A)
                    acc_B, _ = accuracy(pred_B, label_B)
                    preds_all.append(pred_A)
                    preds_all.append(pred_B)
                    labels_all.append(label_A)
                    labels_all.append(label_B)
                    acc_meter.update((acc_A + acc_B) * 0.5)
        kappa_n0, Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, 37)
        print(f'Validation metrics: Kappa={kappa_n0}, F1={Fscd}, OA={acc_meter.avg}, mIoU={IoU_mean}, SeK={Sek}')
        return kappa_n0, Fscd, IoU_mean, Sek, acc_meter.avg

    def _should_validate(self):
        if self.global_iter == 5000:
            return True
        elif self.global_iter < 25000:
            return (self.global_iter % 5000 == 0)
        else:
            return (self.global_iter % 1000 == 0)

    def _save_checkpoint(self, stage_name):
        checkpoint_path = os.path.join(self.model_save_path, f"{stage_name}_checkpoint_iter_{self.global_iter}.pth")
        torch.save(self.deep_model.state_dict(), checkpoint_path)
        print(f"Saved {stage_name} checkpoint at iteration {self.global_iter} to {checkpoint_path}")

    def train_diffusion(self, num_iterations=5000):
        print("Starting Diffusion Pretraining Stage")
        for name, param in self.deep_model.named_parameters():
            if "diffusion" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        diffusion_params = [p for n, p in self.deep_model.named_parameters() if "diffusion" in n and p.requires_grad]
        optimizer = optim.AdamW(diffusion_params, lr=self.lr, weight_decay=self.args.weight_decay)
        # Create a scheduler if desired.
        self.scheduler = StepLR(optimizer, step_size=10000, gamma=0.5)
        
        self.deep_model.train()
        data_iter = iter(self.train_data_loader)
        for i in tqdm(range(num_iterations), desc="Diffusion Training"):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_data_loader)
                data = next(data_iter)
            pre_change_imgs, post_change_imgs, *_ = data
            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            x = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
            batch_size = x.shape[0]
            timesteps = torch.randint(0, 1000, (batch_size,), device=x.device).long()
            noise = torch.randn_like(x)
            x_noisy = self.deep_model.diffusion.q_sample(x, timesteps, noise=noise)
            timesteps = timesteps.float()
            if x_noisy.dtype != torch.float32:
                x_noisy = x_noisy.float()
            predicted_noise, _ = self.deep_model.diffusion.model(x_noisy, timesteps)
            diffusion_loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            diffusion_loss.backward()
            optimizer.step()
            self.scheduler.step()
            
            self.global_iter += 1
            if self.global_iter % 10 == 0:
                print(f"[Diffusion Stage] Iter {self.global_iter}: Loss = {diffusion_loss.item()}")
                self.writer.add_scalar('Loss/Diffusion_Pretrain', diffusion_loss.item(), self.global_iter)
            if self._should_validate():
                val_loss = self.validate_diffusion()
                print(f"[Diffusion Stage] Iter {self.global_iter}: Validation Loss = {val_loss}")
                self.writer.add_scalar('Val/Diffusion', val_loss, self.global_iter)
                self._save_checkpoint("diffusion")
    
    def train_changedecoder(self, num_iterations=20000):
        print("Starting Change Decoder Training Stage")
        for name, param in self.deep_model.named_parameters():
            if "diffusion" in name or "decoder_T1" in name or "decoder_T2" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True  # This should update decoder_bcd and change_attention modules.
        
        changedecoder_params = [p for n, p in self.deep_model.named_parameters() if p.requires_grad]
        optimizer = optim.AdamW(changedecoder_params, lr=self.lr, weight_decay=self.args.weight_decay)
        self.scheduler = StepLR(optimizer, step_size=10000, gamma=0.5)
        
        self.deep_model.train()
        data_iter = iter(self.train_data_loader)
        for i in tqdm(range(num_iterations), desc="Change Decoder Training"):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_data_loader)
                data = next(data_iter)
            pre_change_imgs, post_change_imgs, label_cd, *_ = data
            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            label_cd = label_cd.cuda().long()
            x = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
            # Forward pass – note that the model returns multiple outputs.
            output_1, _, _, _ = self.deep_model(pre_change_imgs, post_change_imgs)
            # Compute change detection loss only.
            ce_loss_cd = ce2_dice1(output_1, label_cd, ignore_index=255)
            lovasz_loss_cd = L.lovasz_softmax(F.softmax(output_1, dim=1), label_cd, ignore=255)
            loss_change = ce_loss_cd + 0.5 * lovasz_loss_cd  # adjust weight as desired.
            
            optimizer.zero_grad()
            loss_change.backward()
            optimizer.step()
            self.scheduler.step()
            
            self.global_iter += 1
            if self.global_iter % 10 == 0:
                print(f"[Change Decoder Stage] Iter {self.global_iter}: Loss = {loss_change.item()}")
                self.writer.add_scalar('Loss/ChangeDecoder', loss_change.item(), self.global_iter)
            if self._should_validate():
                # Use the detection validation for the change detection branch.
                kappa, F1, mIoU, SeK, OA = self.validate_detection()
                self.writer.add_scalar('Metrics/Kappa', kappa, self.global_iter)
                self.writer.add_scalar('Metrics/F1', F1, self.global_iter)
                self.writer.add_scalar('Metrics/mIoU', mIoU, self.global_iter)
                self.writer.add_scalar('Metrics/SeK', SeK, self.global_iter)
                self.writer.add_scalar('Metrics/OA', OA, self.global_iter)
                self._save_checkpoint("changedecoder")
    
    def train_semantic(self, num_iterations=20000):
        """Stage 3: Train the semantic change detection branch.
           Here, freeze diffusion and change decoder; update encoder and semantic decoders."""
        print("Starting Semantic Change Detection Training Stage")
        for name, param in self.deep_model.named_parameters():
            if "diffusion" in name or "decoder_bcd" in name or "change_attention" in name:
                param.requires_grad = False
            else:
                # Update encoder and semantic decoders.
                param.requires_grad = True
        
        semantic_params = [p for n, p in self.deep_model.named_parameters() if p.requires_grad]
        optimizer = optim.AdamW(semantic_params, lr=self.lr, weight_decay=self.args.weight_decay)
        self.scheduler = StepLR(optimizer, step_size=10000, gamma=0.5)
        
        self.deep_model.train()
        data_iter = iter(self.train_data_loader)
        for i in tqdm(range(num_iterations), desc="Semantic Detection Training"):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_data_loader)
                data = next(data_iter)
            pre_change_imgs, post_change_imgs, label_cd, label_clf_t1, label_clf_t2, _ = data
            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            label_cd = label_cd.cuda().long()
            label_clf_t1 = label_clf_t1.cuda().long()
            label_clf_t2 = label_clf_t2.cuda().long()
            # Adjust labels.
            label_clf_t1[label_clf_t1 == 0] = 255
            label_clf_t2[label_clf_t2 == 0] = 255
            output_1, output_semantic_t1, output_semantic_t2, _ = self.deep_model(pre_change_imgs, post_change_imgs)
            # Compute semantic losses.
            ce_loss_clf_t1 = ce2_dice1_multiclass(output_semantic_t1, label_clf_t1)
            ce_loss_clf_t2 = ce2_dice1_multiclass(output_semantic_t2, label_clf_t2)
            lovasz_loss_clf_t1 = L.lovasz_softmax(F.softmax(output_semantic_t1, dim=1), label_clf_t1, ignore=255)
            lovasz_loss_clf_t2 = L.lovasz_softmax(F.softmax(output_semantic_t2, dim=1), label_clf_t2, ignore=255)
            similarity_mask = (label_clf_t1 == 255).float().unsqueeze(1).expand_as(output_semantic_t1)
            similarity_loss = F.mse_loss(F.softmax(output_semantic_t1, dim=1) * similarity_mask, 
                                         F.softmax(output_semantic_t2, dim=1) * similarity_mask, reduction='mean')
            loss_semantic = ce_loss_clf_t1 + ce_loss_clf_t2 + 0.5 * (lovasz_loss_clf_t1 + lovasz_loss_clf_t2) + 0.5 * similarity_loss
            
            optimizer.zero_grad()
            loss_semantic.backward()
            optimizer.step()
            self.scheduler.step()
            
            self.global_iter += 1
            if self.global_iter % 10 == 0:
                print(f"[Semantic Stage] Iter {self.global_iter}: Loss = {loss_semantic.item()}")
                self.writer.add_scalar('Loss/Semantic', loss_semantic.item(), self.global_iter)
            if self._should_validate():
                kappa, F1, mIoU, SeK, OA = self.validate_detection()
                self.writer.add_scalar('Metrics/Kappa', kappa, self.global_iter)
                self.writer.add_scalar('Metrics/F1', F1, self.global_iter)
                self.writer.add_scalar('Metrics/mIoU', mIoU, self.global_iter)
                self.writer.add_scalar('Metrics/SeK', SeK, self.global_iter)
                self.writer.add_scalar('Metrics/OA', OA, self.global_iter)
                self._save_checkpoint("semantic")
    
    def training(self):
        self.train_diffusion(num_iterations=5000)
        self.train_changedecoder(num_iterations=20000)
        self.train_semantic(num_iterations=20000)
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Training on SECOND dataset")
    parser.add_argument('--cfg', type=str, default='/home/songjian/project/RemoteSensing/VMamba/classification/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)

    parser.add_argument('--dataset', type=str, default='SECOND')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default='/data/ggeoinfo/datasets/xBD/train')
    parser.add_argument('--train_data_list_path', type=str, default='/data/ggeoinfo/datasets/xBD/xBD_list/train_all.txt')
    parser.add_argument('--test_dataset_path', type=str, default='/data/ggeoinfo/datasets/xBD/test')
    parser.add_argument('--test_data_list_path', type=str, default='/data/ggeoinfo/datasets/xBD/xBD_list/val_all.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='MambaSCD')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default=0)

    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
