import sys
import os
main_dir = os.path.dirname(os.path.dirname(os.path.dirname((os.path.dirname(__file__)))))
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
from RemoteSensing.changedetection.datasets.make_data_loader import SemanticChangeDetectionDatset, make_data_loader, SemanticChangeDetectionDatset_LandSat
from RemoteSensing.changedetection.utils_func.metrics import Evaluator
from RemoteSensing.changedetection.models.STMambaSCD import STMambaSCD
import RemoteSensing.changedetection.utils_func.lovasz_loss as L
from torch.optim.lr_scheduler import StepLR
from RemoteSensing.changedetection.utils_func.mcd_utils import accuracy, SCDD_eval_all, AverageMeter

from RemoteSensing.changedetection.utils_func.loss import contrastive_loss, ce2_dice1, ce2_dice1_multiclass, SeK_Loss

from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)

        self.deep_model = STMambaSCD(
            output_cd = 2, 
            output_clf = 7 if self.args.dataset == "SECOND" else 10,
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
        self.model_save_path = os.path.join(args.model_param_path, f'CA_spatial_fft_17_7{args.dataset}')
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

        if args.resume is not None:
            self.optim.load_state_dict(torch.load(args.optim_path))
            self.scheduler.load_state_dict(torch.load(args.scheduler_path))

        self.log_dir = os.path.join(main_dir,'saved_models', f'CA_spatial_fft_17_7{args.dataset}')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'logs'))

    def training(self):
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        sek_criterion = SeK_Loss(
            num_classes=7 if self.args.dataset == "SECOND" else 10,  # SECOND dataset classes (exclude non-change)
            non_change_class=0,
            beta=1.5
        ).cuda()
        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            pre_change_imgs, post_change_imgs, label_cd, label_clf_t1, label_clf_t2, _ = data

            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            label_cd = label_cd.cuda().long()
            label_clf_t1 = label_clf_t1.cuda().long()
            label_clf_t2 = label_clf_t2.cuda().long()

            change_mask = (label_cd != 0).float()

            label_clf_t1[label_clf_t1 == 0] = 255
            label_clf_t2[label_clf_t2 == 0] = 255

            output_1, output_semantic_t1, output_semantic_t2 = self.deep_model(pre_change_imgs, post_change_imgs)

            pre_change_imgs = pre_change_imgs.float()
            post_change_imgs = post_change_imgs.float()

            sek_loss_value = sek_criterion(
                output_semantic_t1, 
                output_semantic_t2,
                label_clf_t1,
                label_clf_t2,
                change_mask
            )

            # ================== Auxiliary Losses ==================
            # 1. Semantic segmentation losses
            ce_loss_cd = F.cross_entropy(output_1, label_cd, ignore_index=255)
            ce_loss_clf_t1 = F.cross_entropy(output_semantic_t1, label_clf_t1, ignore_index=255)
            ce_loss_clf_t2 = F.cross_entropy(output_semantic_t2, label_clf_t2, ignore_index=255)
            
            # 2. Boundary refinement
            lovasz_loss_cd = L.lovasz_softmax(F.softmax(output_1, dim=1), label_cd, ignore=255)
            lovasz_loss_clf_t1 = L.lovasz_softmax(F.softmax(output_semantic_t1, dim=1), label_clf_t1, ignore=255)
            lovasz_loss_clf_t2 = L.lovasz_softmax(F.softmax(output_semantic_t2, dim=1), label_clf_t2, ignore=255)
            
            # 3. Temporal consistency 
            similarity_mask = (label_clf_t1 == 255).float().unsqueeze(1)
            similarity_loss = F.mse_loss(
                F.softmax(output_semantic_t1, dim=1) * similarity_mask,
                F.softmax(output_semantic_t2, dim=1) * similarity_mask,
                reduction='mean'
            )

            # ================== Loss Weighting ==================
            weights = {
                'sek': 0,
                'bcd': 1,
                'ce': 1,
                'lovasz': 1,
                'similarity': 0.05
            }
            weights_second = {
                'sek': 0,
                'bcd': 1,
                'ce': 0.5,
                'lovasz': 0.5,
                'similarity': 0.05
            }
            if self.args.dataset == 'SECOND':
                weights = weights_second
            
            SEK_START_ITER = 15000 if self.args.dataset == 'SECOND' else 50000

            if itera + self.args.start_iter > SEK_START_ITER:
                weights['sek'] = 0.5
                weights['bcd'] = 0.4
                weights['ce'] = 0.25
                weights['lovasz'] = 0.25
                weights['similarity'] = 0.025

            total_loss = (
                weights['sek'] * sek_loss_value +
                weights['bcd'] * ce_loss_cd +
                weights['ce'] * (ce_loss_clf_t1 + ce_loss_clf_t2) +
                weights['lovasz'] * (lovasz_loss_clf_t1 + lovasz_loss_clf_t2 + lovasz_loss_cd) +
                weights['similarity'] * similarity_loss
            )

            
            # Backpropagation
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()
            self.scheduler.step()

            if (itera + 1) % 10 == 0:
                print(f'iter is {itera + 1 + self.args.start_iter}, change detection loss is {weights["bcd"] * ce_loss_cd}, '
                      f'classification loss is {weights["ce"] * (ce_loss_clf_t1 + ce_loss_clf_t2) + weights["lovasz"] * (lovasz_loss_clf_t1 + lovasz_loss_clf_t2)}, '
                      f'SeK loss is {1.4 * sek_loss_value}')
                self.writer.add_scalar('Loss/ChangeDetection', weights["bcd"] * ce_loss_cd, itera + 1 + self.args.start_iter)
                self.writer.add_scalar('Loss/Segmentation', 1.4 * sek_loss_value, itera + 1 + self.args.start_iter)
                self.writer.add_scalar('Loss/Classification', weights["ce"] * (ce_loss_clf_t1 + ce_loss_clf_t2) + weights["lovasz"] * (lovasz_loss_clf_t1 + lovasz_loss_clf_t2), itera + 1 + self.args.start_iter)
                self.writer.add_scalar('Loss/Similarity', weights["similarity"] * similarity_loss, itera + 1 + self.args.start_iter)
                self.writer.add_scalar('Loss/Total', total_loss, itera + 1 + self.args.start_iter)
                if ((itera + 1) % 5000 == 0 and (itera + 1 + self.args.start_iter) >= 25000):
                    self.deep_model.eval()
                    kappa_n0, Fscd, IoU_mean, Sek, oa = self.validation()
                    self.writer.add_scalar('Metrics/Kappa', kappa_n0, itera + 1 + self.args.start_iter)
                    self.writer.add_scalar('Metrics/F1', Fscd, itera + 1 + self.args.start_iter)
                    self.writer.add_scalar('Metrics/OA', oa, itera + 1 + self.args.start_iter)
                    self.writer.add_scalar('Metrics/mIoU', IoU_mean, itera + 1 + self.args.start_iter)
                    self.writer.add_scalar('Metrics/SeK', Sek, itera + 1 + self.args.start_iter)
                    if Sek > best_kc:
                        torch.save(self.deep_model.state_dict(),
                                    os.path.join(self.model_save_path, f'{itera + 1 + self.args.start_iter}_model_{Sek:.3f}.pth'))
                        # save adamW optimizer
                        torch.save(self.optim.state_dict(),
                                    os.path.join(self.model_save_path, f'{itera + 1 + self.args.start_iter}_optim_{Sek:.3f}.pth'))
                        torch.save(self.scheduler.state_dict(),
                                    os.path.join(self.model_save_path, f'{itera + 1 + self.args.start_iter}_scheduler_{Sek:.3f}.pth'))
                        best_kc = Sek
                        best_round = [kappa_n0, Fscd, IoU_mean, Sek, oa ]
                    self.deep_model.train()

        print('The accuracy of the best round is ', best_round)
        self.writer.close()

    def validation(self):
        print('---------starting evaluation-----------')
        dataset = SemanticChangeDetectionDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')

        if self.args.dataset == 'LandSat':
            dataset = SemanticChangeDetectionDatset_LandSat(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')

        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
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


                # input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
                output_1, output_semantic_t1, output_semantic_t2 = self.deep_model(pre_change_imgs, post_change_imgs)

                labels_cd = labels_cd.cpu().numpy()
                labels_A = labels_clf_t1.cpu().numpy()
                labels_B = labels_clf_t2.cpu().numpy()

                change_mask = torch.argmax(output_1, axis=1).cpu().numpy()

                preds_A = torch.argmax(output_semantic_t1, dim=1).cpu().numpy()
                preds_B = torch.argmax(output_semantic_t2, dim=1).cpu().numpy()

                preds_A[change_mask == 0] = 0
                preds_B[change_mask == 0] = 0

                if itera % 100 == 0:
                    print(f'iter is {itera}')

                for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                    acc_A, valid_sum_A = accuracy(pred_A, label_A)
                    acc_B, valid_sum_B = accuracy(pred_B, label_B)
                    preds_all.append(pred_A)
                    preds_all.append(pred_B)
                    labels_all.append(label_A)
                    labels_all.append(label_B)
                    acc = (acc_A + acc_B) * 0.5
                    acc_meter.update(acc)

        kappa_n0, Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, 7 if self.args.dataset == "SECOND" else 10)
        print(f'Kappa coefficient rate is {kappa_n0}, F1 is {Fscd}, OA is {acc_meter.avg}, '
              f'mIoU is {IoU_mean}, SeK is {Sek}')
        
        return kappa_n0, Fscd, IoU_mean, Sek, acc_meter.avg

