from pathlib import Path
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('/home/undergrad/AyChihuahua/SAM/RGB_only_v0')
from dst.extract_data_tools.DSEC.provider import DatasetProvider, DatasetImageProvider



def DSECEvent(dsec_dir, nr_events_data=1, delta_t_per_data=50, nr_events_window=100000,
              augmentation=False, mode='train', event_representation='voxel_grid', nr_bins_per_data=5,
              require_paired_data=False, separate_pol=False, normalize_event=False, semseg_num_classes=11,
              fixed_duration=False, random_crop=False):
    """
    Creates an iterator over the EventScape dataset.

    :param root: path to dataset root
    :param height: height of dataset image
    :param width: width of dataset image
    :param nr_events_window: number of events summed in the sliding histogram
    :param augmentation: flip, shift and random window start for training
    :param mode: 'train', 'test' or 'val'
    """
    dsec_dir = Path(dsec_dir)
    assert dsec_dir.is_dir()

    dataset_provider = DatasetProvider(dsec_dir, mode, event_representation=event_representation,
                                       nr_events_data=nr_events_data, delta_t_per_data=delta_t_per_data,
                                       nr_events_window=nr_events_window, nr_bins_per_data=nr_bins_per_data,
                                       require_paired_data=require_paired_data, normalize_event=normalize_event,
                                       separate_pol=separate_pol, semseg_num_classes=semseg_num_classes,
                                       augmentation=augmentation, fixed_duration=fixed_duration, random_crop=random_crop)
    if mode == 'train':
        train_dataset = dataset_provider.get_train_dataset()
        print("[DESCEvent]: Found %s segmentation masks for split train" % (train_dataset.__len__()))
        return train_dataset
    else:
        val_dataset = dataset_provider.get_val_dataset()
        print("[DESCEvent]: Found %s segmentation masks for split test" % (val_dataset.__len__()))
        return val_dataset


def DSECImage(dsec_dir, augmentation=False, mode='train', semseg_num_classes=11, random_crop=False):
    """
    Creates an iterator over the EventScape dataset.

    :param root: path to dataset root
    :param height: height of dataset image
    :param width: width of dataset image
    :param augmentation: flip, shift and random window start for training
    :param mode: 'train', 'test' or 'val'
    """
    dsec_dir = Path(dsec_dir)
    assert dsec_dir.is_dir()

    dataset_provider = DatasetImageProvider(dsec_dir, mode, semseg_num_classes=semseg_num_classes,
                                            augmentation=augmentation, random_crop=random_crop)
    if mode == 'train':
        train_dataset = dataset_provider.get_train_dataset()
        print("[DESCImage]: Found %s segmentation masks for split train" % (train_dataset.__len__()))
        return train_dataset
    else:
        val_dataset = dataset_provider.get_val_dataset()
        print("[DESCImage]: Found %s segmentation masks for split test" % (val_dataset.__len__()))
        return val_dataset


if __name__ == "__main__":
    # provider已做更改，现在是train_events和train_images，test同理。
    # dsec_seg_path = '/NVME_id0/DVS_Data/DSEC/'
    dsec_seg_path = '/DataHDD0/lk_temp_dsec/'
    # """
    dsec_ev = DSECEvent(dsec_dir=dsec_seg_path, mode='train', nr_events_data=1, nr_bins_per_data=3,
                        fixed_duration=False, augmentation=True, random_crop=True)
    dataloader = DataLoader(DSECEvent)
    # i = 1
    # for idx, (event, label) in enumerate(dataloader):
    #     print(i)
    #     print(event.shape, label.shape)
    #     i += 1
    event, label = dsec_ev[100]
    print(event.shape, label.shape) # 不加dataloader torch.Size([3, 448, 448]) torch.Size([448, 448])
    # """

    """
    dsec_img = DSECImage(dsec_dir=dsec_seg_path, mode='val', augmentation=False, random_crop=False)
    # i = 1
    # for event, label in dsec_img:
    #     print(i)
    #     print(event.shape, label.shape)
    #     i += 1
    img, label = dsec_img[100]
    print(img.shape, label.shape)   # torch.Size([3, 440, 640]) torch.Size([440, 640])
    """
