
import torch
from mmengine import Config
import argparse
import torch.nn as nn
import torch.nn.functional as F
import warnings
import sys
import os

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])  # 解决no module named问题
sys.path.append(root_path)
from mmseg.models.builder import build_segmentor
from model import mix_transformer
from mmseg.models.decode_heads import segformer_head


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    # parser.add_argument('config', help='train config file path')
    # parser.add_argument('--work-dir', help='the dir to save logs and models')
    # parser.add_argument(
    #     '--load-from', help='the checkpoint file to load weights from')
    # parser.add_argument(
    #     '--resume-from', help='the checkpoint file to resume from')
    # parser.add_argument(
    #     '--no-validate',
    #     action='store_true',
    #     help='whether not to evaluate the checkpoint during training')
    # group_gpus = parser.add_mutually_exclusive_group()
    # group_gpus.add_argument(
    #     '--gpus',
    #     type=int,
    #     help='number of gpus to use '
    #     '(only applicable to non-distributed training)')
    # group_gpus.add_argument(
    #     '--gpu-ids',
    #     type=int,
    #     nargs='+',
    #     help='ids of gpus to use '
    #     '(only applicable to non-distributed training)')
    # parser.add_argument('--seed', type=int, default=None, help='random seed')
    # parser.add_argument(
    #     '--deterministic',
    #     action='store_true',
    #     help='whether to set deterministic options for CUDNN backend.')
    # parser.add_argument(
    #     '--options', nargs='+', action=DictAction, help='custom options')
    # parser.add_argument(
    #     '--launcher',
    #     choices=['none', 'pytorch', 'slurm', 'mpi'],
    #     default='none',
    #     help='job launcher')
    # parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def seg_512_b2_encoder(): 
    return mix_transformer.mit_b2()


def seg_512_b2_decoder():  
    norm_cfg = dict(type='BN', requires_grad=True)
    decode_head = dict(
        # type='SegFormerHead',
        # type='MLPHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    # print(*decode_head)
    return segformer_head.SegFormerHead(**decode_head)


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class EncoderDecoder(nn.Module):

    def __init__(self,
                 neck=None,
                 train_cfg=dict(),
                 test_cfg=dict(mode='whole'),
                 pretrained=None):
        super(EncoderDecoder, self).__init__()
        backbone = mix_transformer.mit_b0()  # b0
        self.backbone = backbone
        if neck is not None:
            pass
            # self.neck = builder.build_neck(neck)
        self._init_decode_head()
        # self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        
 

    # b0的设置
    def _init_decode_head(self, ):
        # model training and testing settings
        norm_cfg = dict(type='BN', requires_grad=True)
        decode_head = dict(
            # type='SegFormerHead',
            in_channels=[32, 64, 160, 256],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=128,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            decoder_params=dict(embed_dim=256),
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        decode_head2 = dict(
            # type='SegFormerHead',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=128,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            decoder_params=dict(embed_dim=512),
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        self.decode_head = segformer_head.SegFormerHead(**decode_head)
        self.decode_head2 = segformer_head.SegFormerHead(**decode_head2)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def extract_feat(self, img, x1, x2, x3, x4):
        """Extract features from images."""
       
        x, x_fused = self.backbone(img, x1, x2, x3, x4)
        print("haha")
    
        return x, x_fused

    def encode_decode(self, img, x1, x2, x3, x4): 
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x_original, x_fused = self.extract_feat(img, x1, x2, x3, x4)
       
        out_original = self.decode_head(x_original)
        out_fused = self.decode_head2(x_fused)
   
        return out_original, out_fused

    # TODO refactor
    def slide_inference(self, img, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                # size=img_meta[0]['ori_shape'][:2],    
                size=img[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, x1, x2, x3, x4, rescale):
        """Inference with full image."""
        # input:  torch.Size([3, 352, 512])
        seg_logit_original, seg_logit_fused = self.encode_decode(img, x1, x2, x3, x4)

        if rescale:
            seg_logit_original = resize(
                seg_logit_original,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
            seg_logit_fused = resize(
                seg_logit_fused,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        # print('rescale之后的seg_logit形状：',seg_logit) # size=(8, 11, 352, 512)

        return seg_logit_original, seg_logit_fused

    def forward(self, img, x1, x2, x3, x4, mode, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert mode in ['slide', 'whole']
        # for tensor in img:
        # print('whole_inference之前tensor的形状是：',tensor.shape) # torch.Size([3, 352, 512])
        # ori_shape = img.shape[2:]
        if mode == 'slide':
            seg_logit = self.slide_inference(img, rescale)
        else:
            
            seg_logit_original, seg_logit_fused = self.whole_inference(img, x1, x2, x3, x4, rescale)
         
        return seg_logit_original, seg_logit_fused

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred


def main():
    args = parse_args()
    # args.config = '/home/lk/MyProjects/RGB_only_v0/local_configs/segformer/B2/segformer.b2.512x512.ade.160k.py'
    args.config = '/home/lk/MyProjects/RGB_only_v0/local_configs/segformer/B0/segformer.b0.512x512.ade.160k.py'

    cfg = Config.fromfile(args.config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))


if __name__ == '__main__':
    inp = torch.rand((16, 3, 512, 512))
    GT = torch.rand((16, 512, 512))
   
    segformer = EncoderDecoder()
    # print(segformer)
    path_weights = "/ESEG/ckpt/segformer.b0.512x512.ade.160k.pth"
    # path_weights = "/home/lk/MyProjects/RGB_only_v0/ckpt/mit_b2_20220624-66e8bf70.pth"
    model_pretrained = torch.load(path_weights)
    print(model_pretrained['meta'].keys())
    pretrained_weights = model_pretrained["state_dict"]
    print(pretrained_weights.keys())
    newParams = segformer.state_dict().copy()
    # for (name, param), (name_pretrain, param_pretrain) in zip(newParams.items(), pretrained_weights.items()):
    #     print(name, param.shape, name_pretrain, param_pretrain.shape)
    #     newParams[name] = pretrained_weights[name_pretrain]
    print("-----------------------------------------------------------------------------------")
    for (name, param) in newParams.items():
        print(name, param.shape, pretrained_weights[name].shape)
        # newParams[name] = pretrained_weights[name]
    # print(segformer.decode_head.linear_c1.proj.weight)
    segformer.load_state_dict(newParams)
    segformer.decode_head.linear_pred = nn.Conv2d(768, 11, kernel_size=1)  # 将这个卷积核作为最后的线性预测头
    print(segformer.decode_head.linear_pred.weight.shape)
    # print(segformer)
    pred = segformer(inp, mode="whole", rescale=True)
    print(pred.shape)
    # print(segformer.decode_head.linear_c1.proj.weight)
    # print(segformer.backbone)
    # pred = segformer.forward(inp)
    # print(pred.shape)

    norm_cfg = dict(type='SyncBN', requires_grad=True)
    decode_head = dict(
        # type='SegFormerHead',
        # type='MLPHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    decode_head = segformer_head.SegFormerHead(**decode_head)
    print(decode_head)
