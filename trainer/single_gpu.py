import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2]) 
sys.path.append(root_path)
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, PolynomialLR
import tqdm
import numpy as np
import random
import time
import os.path as osp
import sklearn.metrics as metrics
import warnings
from torch.utils.data import Dataset, DataLoader
# from lightly.loss.ntx_ent_loss import NTXentLoss
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# 这个是单卡的代码
# from model.mix_transformer import mit_b0
from model.segformer_build import EncoderDecoder
from utils.utils_func import IOStream
from losses.loss_func import TaskLoss
from dst.dsec_dataset import DSECEvent, DSECImage
from utils.metrics import MetricsSemseg

####################
"""VERSION_BACKUP"""
####################
proj_home_path = "/home/lk/MyProjects/RGB_only_v0"
warnings.filterwarnings("ignore")

def get_local_dir(path):
    return os.path.join(proj_home_path, path)


class Trainer():

    def __init__(self, args):

        self.args = args
        self.initTime = self.get_local_time()
        self.save_to_dir = self.get_save_dir()
        self.io = IOStream(self.save_to_dir + '/run.log')
        self.io.cprint(str(self.args))

        """Random seed setting"""
        SEED = self.args.seed
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        """Training device defining"""
        self.args.cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        if args.cuda:
            self.io.cprint(
                'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(
                    torch.cuda.device_count()) + ' devices')
            torch.cuda.manual_seed(self.args.seed)
            torch.backends.cudnn.deterministic = True
        else:
            self.io.cprint('Using CPU')


        """Dataloader init"""
         # dsec_seg_path = '/NVME_id0/DVS_Data/DSEC/'    # rgb
        dsec_seg_path = '/DataHDD0/lk_temp_dsec/'       # event
        # self.train_loader = DataLoader(DSECImage(dsec_dir=dsec_seg_path, mode='train', augmentation=True, random_crop=True),
        #                                num_workers=self.args.num_workers,
        #                                batch_size=self.args.batch_size,
        #                                shuffle=self.args.shuffle,
        #                                drop_last=self.args.drop_last)

        # self.test_loader = DataLoader(DSECImage(dsec_dir=dsec_seg_path, mode='val', augmentation=False, random_crop=False),
        #                               num_workers=self.args.num_workers,
        #                               batch_size=self.args.test_batch_size,
        #                               shuffle=False,
        #                               drop_last=False)
        train_dataset = DSECEvent(dsec_dir=dsec_seg_path,delta_t_per_data=50, mode='train', nr_events_data=1, nr_bins_per_data=3,
                        fixed_duration=True, augmentation=True, random_crop=True)
        test_dataset = DSECEvent(dsec_dir=dsec_seg_path,delta_t_per_data=50, mode='val', nr_events_data=1, nr_bins_per_data=3,
                        fixed_duration=True, augmentation=False, random_crop=False)
       
        self.train_loader = DataLoader( train_dataset,      
                        num_workers=self.args.num_workers,
                                       batch_size=self.args.batch_size,
                                       shuffle=self.args.shuffle,
                                       drop_last=self.args.drop_last)

        self.test_loader = DataLoader( test_dataset,
                        num_workers=self.args.num_workers,
                                       batch_size=self.args.test_batch_size,
                                       shuffle=False,
                                       drop_last=False)

        """ model setting"""
        self.model = EncoderDecoder()

        """Loading pretrained weights"""
        path_weights = "/home/lk/MyProjects/RGB_only_v0/ckpt/segformer.b0.512x512.ade.160k.pth"
        model_pretrained = torch.load(path_weights)
        print(model_pretrained['meta'].keys())
        pretrained_weights = model_pretrained["state_dict"]
        newParams = self.model.state_dict().copy()
        for (name, param) in newParams.items():
            newParams[name] = pretrained_weights[name]
        # model_dict = self.model.state_dict()
        # pretrained_dict = {k: v for k, v in model_pretrained.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # self.model.load_state_dict(model_dict)       
        self.model.load_state_dict(newParams)
        # self.model.decode_head.linear_pred = nn.Conv2d(768, 11, kernel_size=1) # b2
        self.model.decode_head.linear_pred = nn.Conv2d(256, 11, kernel_size=1) # b0


        """Parallel settings"""
        self.model = nn.DataParallel(self.model.to(self.device))
        self.model = torch.compile(self.model)
        outstr = "Let's use " + str(torch.cuda.device_count()) + " GPUs!"
        self.io.cprint(outstr)


        if self.args.train_from_checkpoint:
            model_saved = torch.load(self.args.model_path)
            print(model_saved["best_score"], model_saved["Epoch_num"])
            pretrainParams = model_saved["Weights"]
            newParams = self.model.state_dict().copy()
            for (name, param), (name_pretrain, param_pretrain) in zip(newParams.items(), pretrainParams.items()):
                newParams[name] = pretrainParams[name_pretrain]
            self.model.load_state_dict(newParams)

        if self.args.use_sgd:
            self.io.cprint("Use SGD")
            self.opt = optim.SGD([{'params': self.model.parameters()}], lr=self.args.lr, momentum=self.args.momentum,
                                 weight_decay=1e-4)
        else:
            self.io.cprint("Use AdamW")
            self.opt = optim.AdamW([{'params': self.model.parameters(), 'lr': self.args.lr}], weight_decay=0.02, betas= (0.8, 0.99))

        if self.args.scheduler == 'cos':
            self.scheduler = CosineAnnealingLR(self.opt, self.args.num_epochs,
                                               eta_min=1e-4)
        elif self.args.scheduler == 'step':
            self.scheduler = StepLR(self.opt, step_size=30, gamma=0.5)
        elif self.args.scheduler == 'poly':
            self.scheduler = PolynomialLR(self.opt, total_iters=60, power=1.0, verbose=True)

        """Metrics and Losses setting"""
        self.matrics_stat = MetricsSemseg(num_classes = 11, ignore_label = 255,
                                        class_names = ['background', 'building', 'fence', 'person', 'pole', 'road',
                                                       'sidewalk', 'vegetation', 'car', 'wall', 'traffic sign'])

        self.criterion = TaskLoss(losses=['dice', 'cross_entropy'],
                                  gamma=2.0, num_classes=11, alpha=None, weight=None, ignore_index=255)

    ####################
    """Train (single epoch)"""

    ####################
    def train_pass(self, epoch):
        for param_group in self.opt.param_groups:
            lr_str = "\nlr is: {}".format(param_group['lr'])
            self.io.cprint(lr_str)
        train_loss = 0.0
        target_loss = 0.0
        count = 0.0
        self.model.train()
        train_pred = []
        teach_pred = []
        train_true = []
        for idx, (img, label) in tqdm.tqdm(enumerate(self.train_loader)):
            img = img.float().to(self.device)
            label = label.to(self.device).squeeze()
            batch_size = img.size()[0]

            """model forwarding"""
            logits_pred = self.model(img, mode="whole", rescale=True)


            """Losses calculating"""
            loss = self.criterion(logits_pred, label)

            """Back-propagating"""
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            """Statistics"""
            count += batch_size
            train_loss += loss.item() * batch_size
            label_pred = torch.argmax(logits_pred.squeeze(), dim=1)
            self.matrics_stat.update_batch(
                label_pred.unsqueeze(1).detach(),
                label.unsqueeze(1).detach())



        """Statistics print"""
        metrics_summary = self.matrics_stat.get_metrics_summary()
        outstr = 'Train %d, ' \
                 '\nloss: %.2f, mAcc: %.2f, mIoU: %.2f' % (epoch, train_loss * 1.0 / count,
                                                           metrics_summary['acc'],
                                                           metrics_summary['mean_iou'])
        self.io.cprint(outstr)

    ####################
    """Eval (single epoch)"""

    ####################
    def eval_pass(self, epoch, best_score):
        self.model.eval()
        test_loss = 0.0
        count = 0.0
        pred_kd_loss = 0.0
        mid_kd_loss = 0.0
        target_loss = 0.0
        test_pred = []
        test_true = []
        test_teach_pred = []
        with torch.no_grad():
            for idx, (img, label) in tqdm.tqdm(enumerate(self.test_loader)):
                img = img.float().to(self.device)
                label = label.to(self.device).squeeze()
                batch_size = img.size()[0]

                """Inference"""

                """model forwarding"""
                logits_pred = self.model(img, mode="whole", rescale=True)
               

                """Losses calculating"""
                loss = self.criterion(logits_pred, label)

                """Statistics"""
                count += batch_size
                test_loss += loss.item() * batch_size
                label_pred = torch.argmax(logits_pred.squeeze(), dim=1)
                self.matrics_stat.update_batch(
                    label_pred.unsqueeze(1).detach(),
                    label.unsqueeze(1).detach())

        """Statistics print"""
        metrics_summary = self.matrics_stat.get_metrics_summary()
        outstr = 'Test %d, ' \
                 '\nloss: %.2f, mAcc: %.2f, mIoU: %.2f' % (epoch, test_loss * 1.0 / count,
                                                           metrics_summary['acc'],
                                                           metrics_summary['mean_iou'])
        self.io.cprint(outstr)

        if metrics_summary['mean_iou'] >= best_score:
            best_score = metrics_summary['mean_iou']
            self.save_model(epoch=epoch, best_score=best_score)
            str = "Best Performance is: {}".format(best_score)
            self.io.cprint(str)

        return best_score

    def train(self):

        best_score = 0

        for epoch in range(self.args.num_epochs):

            self.matrics_stat.reset()
            self.train_pass(epoch)

            self.matrics_stat.reset()
            best_score = self.eval_pass(epoch, best_score)


            """Learning rate modurating"""
            if self.args.scheduler == 'cos':
                self.scheduler.step()
            elif self.args.scheduler == 'step':
                if self.opt.param_groups[0]['lr'] > 1e-6:
                    self.scheduler.step()
                if self.opt.param_groups[0]['lr'] < 1e-6:
                    for param_group in self.opt.param_groups:
                        param_group['lr'] = 1e-6
            else:
                self.scheduler.step()

    def eval(self):
        self.eval_pass(epoch=0, best_score=None)

    def get_save_dir(self):
        path = osp.join('./log_file', self.args.exp_name)
        dir_path = osp.join(path, self.initTime)
        if not osp.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path

    def save_model(self, epoch, best_score=None):
        netDict = self.model.state_dict()
        saveTo = self.save_to_dir

        if best_score is None:
            pass
        else:
            torch.save({
                "Weights": netDict,
                "best_score": best_score,
                "Epoch_num": epoch,
            }, osp.join(saveTo, "BestModel.pth"))

    def get_local_time(self):
        return time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())



if __name__ == '__main__':

    # -------------------------------------------------------------------------------------------------------------------- #
    """Training settings"""
    # -------------------------------------------------------------------------------------------------------------------- #

    parser = argparse.ArgumentParser(description='Seg_RGB_DSEC')

    # """Universal settings"""
    parser.add_argument('--exp_name', type=str, default='RGB_SEG_DSEC', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    #
    # """Runing modes"""
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    # parser.add_argument('--frame_based_models', type=str, default='', metavar='N',
    #                     help='Pretrained weights of the frame-based model')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--train_from_checkpoint', type=bool, default=False,
                        help='train network from check point')
    parser.add_argument('--num_epochs', type=int, default=60, metavar='N',
                        help='number of episode to train ')
    #
    # """Dataset settings"""
    # parser.add_argument('--training_path', type=str, default='', metavar='N',
    #                     help='path to training data')
    # parser.add_argument('--validation_path', type=str, default='', metavar='N',
    #                     help='path to validation data')
    # parser.add_argument('--num_points', type=int, default=2048,
    #                     help='num of points to inference')
    # parser.add_argument('--sensor_size', type=int, default=[180, 240],
    #                     help='sensor size of cameras')
    # parser.add_argument('--voxel_size', type=int, default=[10, 10, 25 * 1e3],
    #                     help='voxelizating sizes')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num_workers')
    # parser.add_argument('--ifbin', type=int, default=False, metavar='ifbin',
    #                     help='If file type is .bin')
    # parser.add_argument('--augmentation', type=bool, default=True,
    #                     help='If augmentation, e.g., shifting, random interval etc)')
    parser.add_argument('--shuffle', type=int, default=True, metavar='shuffle',
                        help='If shuffle')
    parser.add_argument('--drop_last', type=int, default=False, metavar='drop_last',
                        help='If drop_last')
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='test_batch_size',
                        help='Size of batch)')
    #
    # """Model settings"""
    # parser.add_argument('--model_t', type=str, default='resnet18', metavar='N',
    #                     choices=['resnet34', 'resnet18'], help='Model to use')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # parser.add_argument('--topk_num', type=int, default=20, metavar='N',
    #                     help='Num of nearest neighbors to use')
    # parser.add_argument('--num_classes', type=int, default=101, metavar='N',
    #                     help='Dataset categories')
    # parser.add_argument('--feat_dim', type=int, default=[25, 64, 64, 128], metavar='N',
    #                     help='Dimensions of input feats')
    # parser.add_argument('--ifpretrain', type=bool, default=True,
    #                     help='if pretrained on ImageNet')
    # parser.add_argument('--ifmultibranch', type=bool, default=True,
    #                     help='if multi-branch prediction')
    # parser.add_argument('--ifnokd', type=bool, default=False,
    #                     help='if multi-branch prediction')
    # parser.add_argument('--num_channel', type=int, default=6, metavar='N',
    #                     help='input channel')
    #
    # """Optimization settings"""
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.00006, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
    #                     help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='poly', metavar='N',
                        choices=['cos', 'step', 'poly'],
                        help='Scheduler to use, [cos, step, poly]')
    #
    args = parser.parse_args()
    #
    trainer = Trainer(args)
    if not args.eval:
        trainer.train()
    # else:
    #     trainer.eval()
