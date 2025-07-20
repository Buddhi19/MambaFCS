import sys
import os
main_dir = os.path.dirname(os.path.dirname(os.path.dirname((os.path.dirname(__file__)))))
sys.path.append(main_dir)

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
from RemoteSensing.changedetection.datasets.make_data_loader import ChangeDetectionDatset, make_data_loader
from RemoteSensing.changedetection.utils_func.metrics import Evaluator
from RemoteSensing.changedetection.models.MambaBCD import STMambaBCD
from RemoteSensing.changedetection.utils_func.loss import ce2_dice1

import RemoteSensing.changedetection.utils_func.lovasz_loss as L

from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)

        self.evaluator = Evaluator(num_class=2)

        self.deep_model = STMambaBCD(
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
        
        self.model_save_path = os.path.join(args.model_param_path, f"{args.model_saving_name}")

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

        self.log_dir = os.path.join(main_dir,'saved_models', f'{args.model_saving_name}')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'logs'))

    def training(self):
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            pre_change_imgs, post_change_imgs, labels, _ = data

            pre_change_imgs = pre_change_imgs.cuda().float()
            post_change_imgs = post_change_imgs.cuda()
            labels = labels.cuda().long()

            output_1 = self.deep_model(pre_change_imgs, post_change_imgs)

            self.optim.zero_grad()
            
            ce_loss_1 = ce2_dice1(output_1, labels)

            lovasz_loss = L.lovasz_softmax(F.softmax(output_1, dim=1), labels, ignore=255)
            

            final_loss = ce_loss_1 + 0.5*lovasz_loss

            final_loss.backward()
            self.optim.step()

            self.writer.add_scalar('CDLoss/train', final_loss.item(), itera + 1)

            if (itera + 1) % 10 == 0:
                print(f'iter is {itera + 1}, overall loss is {final_loss}')
                if (itera + 1) % 500 == 0:
                    self.deep_model.eval()
                    rec, pre, oa, f1_score, iou, kc = self.validation()
                    self.writer.add_scalar('CDMetrics/Recall', rec, itera + 1)
                    self.writer.add_scalar('CDMetrics/Precision', pre, itera + 1)
                    self.writer.add_scalar('CDMetrics/OA', oa, itera + 1)
                    self.writer.add_scalar('CDMetrics/F1_score', f1_score, itera + 1)
                    self.writer.add_scalar('CDMetrics/IoU', iou, itera + 1)
                    self.writer.add_scalar('CDMetrics/Kappa', kc, itera + 1)
                    if kc > best_kc:
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'{itera + 1}_model_{kc}.pth'))
                        best_kc = kc
                        best_round = [rec, pre, oa, f1_score, iou, kc]
                    self.deep_model.train()

        print('The accuracy of the best round is ', best_round)
        self.writer.close()

    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator.reset()
        dataset = ChangeDetectionDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=16, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels, _ = data
                pre_change_imgs = pre_change_imgs.cuda().float()
                post_change_imgs = post_change_imgs.cuda()
                labels = labels.cuda().long()

                output_1 = self.deep_model(pre_change_imgs, post_change_imgs)

                output_1 = output_1.data.cpu().numpy()
                output_1 = np.argmax(output_1, axis=1)
                labels = labels.cpu().numpy()

                self.evaluator.add_batch(labels, output_1)
                
        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        print(f'Racall rate is {rec}, Precision rate is {pre}, OA is {oa}, '
              f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')
        return rec, pre, oa, f1_score, iou, kc

