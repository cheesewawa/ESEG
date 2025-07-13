import numpy
import numpy as np
import torch
from functools import partial


def generate_input_representation(events, event_representation, shape, nr_temporal_bins=5, separate_pol=True):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {-1, 1}. x and y correspond to image
    coordinates u and v.
    """
    if event_representation == 'histogram':
        return generate_event_histogram(events, shape)
    elif event_representation == 'voxel_grid':
        return generate_voxel_grid(events, shape, nr_temporal_bins, separate_pol)


def generate_event_histogram(events, shape):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {-1, 1}. x and y correspond to image
    coordinates u and v.
    """
    events = events[events[:, 0] >= 0]
    events = events[events[:, 0] <= shape[1]]
    events = events[events[:, 1] >= 0]
    events = events[events[:, 1] <= shape[0]]
    height, width = shape
    x, y, t, p = events.T
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    p[p == 0] = -1  # polarity should be +1 / -1
    img_pos = np.zeros((height * width,), dtype="float32")
    img_neg = np.zeros((height * width,), dtype="float32")

    np.add.at(img_pos, x[p == 1] + width * y[p == 1], 1)
    np.add.at(img_neg, x[p == -1] + width * y[p == -1], 1)

    histogram = np.stack([img_neg, img_pos], 0).reshape((2, height, width))

    return histogram

def flatten_and_unflatten(x,shape):
    original_shape = shape
    # print('original_shape is:',original_shape)
    # print('x shape:',x.shape)
    # return x.flatten(), partial(np.reshape, newshape=original_shape)
    return x.flatten(),x.reshape((3, 480, 640))

def normalize_voxel_grid(events):
    """Normalize event voxel grids"""
    nonzero_ev = (events != 0)
    num_nonzeros = nonzero_ev.sum()
    if num_nonzeros > 0:
        # compute mean and stddev of the **nonzero** elements of the event tensor
        # we do not use PyTorch's default mean() and std() functions since it's faster
        # to compute it by hand than applying those funcs to a masked array
        mean = events.sum() / num_nonzeros
        stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero_ev.float()
        events = mask * (events - mean) / stddev

    return events


def generate_voxel_grid(events, shape, nr_temporal_bins, separate_pol=True):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param nr_temporal_bins: number of bins in the temporal axis of the voxel grid
    :param shape: dimensions of the voxel grid
    """
    height, width = shape

    assert(events.shape[1] == 4)
    assert(nr_temporal_bins > 0)        # nr_temporal_bins代表着最后生成tensor的channel数目。
    assert(width > 0)
    assert(height > 0)

    # voxel_grid_positive = np.zeros((nr_temporal_bins, height, width), np.float64).ravel()
    # voxel_grid_negative = np.zeros((nr_temporal_bins, height, width), np.float64).ravel()
    voxel_grid = torch.zeros(nr_temporal_bins,
                             height,
                             width,
                             dtype=torch.float32,
                             device='cpu')
    voxel_grid_flat, unflatten = flatten_and_unflatten(voxel_grid,shape)
    # normalize the event timestamps so that they lie between 0 and num_bins
    event_tensor = torch.from_numpy(events)
    last_stamp = events[-1, 2]
    first_stamp = events[0, 2]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    # xs = events[:, 0].astype(np.int32)    # 原方法
    # ys = events[:, 1].astype(np.int32)
    # ts = (nr_temporal_bins - 1) * (events[:, 2] - first_stamp) / deltaT
    # pols = events[:, 3] # 极性信息
    # pols[pols == 0] = -1  # polarity should be +1 / -1
    # tis = ts.astype(np.float32)
    # dts = ts - tis
    # vals_left = np.abs(pols) * (1.0 - dts)  #改成int64就全黑了
    # vals_right = np.abs(pols) * dts
    xs = event_tensor[:, 0]
    ys = event_tensor[:, 1]
    ts = (nr_temporal_bins - 1) * (event_tensor[:, 2] - first_stamp) / deltaT
    pols = event_tensor[:, 3].float() # 极性信息
    pols[pols == 0] = -1  # polarity should be +1 / -1
    tis = ts.float()
    
    # print('dts:',dts)   # dts: [0. 0. 0. ... 0. 0. 0.]
    # print(dts.dtype,ts.dtype)   # float64
    left_t, right_t = tis.floor(), tis.floor() + 1
    left_x, right_x = xs.floor(), xs.floor() + 1
    left_y, right_y = ys.floor(), ys.floor() + 1
    # print('left_x',left_x)
   

    # print(vals_left.dtype)  # vals_left是float64
    # vals_left = np.floor(np.abs(pols) * (1.0 - dts))  #改成int64就全黑了
    # vals_right = np.floor((np.abs(pols) * dts)  ) + 1 # 改成floor就变成蓝色的了。。还是带网格
    # print('2:',vals_right)
    # print('vals_left.dtype',vals_left.dtype)  # float64
    pos_events_indices = pols == 1
    # """ voxelization的计算方式
    for lim_x in [left_x, right_x]:
        for lim_y in [left_y, right_y]:
            for lim_t in [left_t, right_t]:
                mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x <= width - 1) \
                       & (lim_y <= height - 1) & (lim_t <= nr_temporal_bins - 1)

                # 在这里转换为long，否则掩码计算不正确
                lin_idx = lim_x.long() \
                          + lim_y.long() * width \
                          + lim_t.long() * width * height
                # print('数据类型：',lin_idx.dtype)   # int64

                weight = pols * (1 - (lim_x - xs).abs()) * (1 - (lim_y - ys).abs()) * (1 - (lim_t - tis).abs())
                # voxel_grid_flat[lin_idx[mask]] += weight[mask]
                # new_voxel_grid_flat = voxel_grid_flat
                new_voxel_grid_flat = voxel_grid_flat.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())
                
    # print(new_voxel_grid_flat.shape)    # (921600,)
    
    voxel_grid = new_voxel_grid_flat.reshape((3, 480, 640))
    voxel_grid_np = voxel_grid.numpy()
    # print('voxel:',voxel_grid.shape)        # 改成这个方法，还是有错误，还是带网格的... # (3, 480, 640)
    # """



    """
    xs[valid_indices_pos] + ys[valid_indices_pos] * width +
              tis[valid_indices_pos] * width * height
    这一段就是类似于下面   的操作。
     # xs+ys+tis就是index，要把vals_left添加到voxel_grid_positive的index处。
    lin_idx = lim_x.long() \
                          + lim_y.long() * event_sequence._image_width \
                          + lim_t.long() * event_sequence._image_width * event_sequence._image_height
    """
    """
    # Positive Voxels Grid
    # 如果把以下改成<= xxx-1，图像会出现紫红色，仍带网格
    valid_pos = (xs < width) & (xs >= 0) & (ys < height) & (ys >= 0) & (ts >= 0) & (ts < nr_temporal_bins)
    # print('valid_pos is :',valid_pos)  [ True  True  True ...  True  True  True]
    valid_indices_pos = np.logical_and(tis < nr_temporal_bins, pos_events_indices)
    # print('valid_indices_pos is:',valid_indices_pos) [False  True  True ... False  True  True]
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)
    # print( 'ys[valid_indices_pos] * width is:', ys[valid_indices_pos] * width )
    index_pos = (xs[valid_indices_pos].astype(np.int64)) + (ys[valid_indices_pos].astype(np.int64)) * width + \
    (tis[valid_indices_pos].astype(np.int64)) * width * height
    np.add.at(voxel_grid_positive,index_pos, vals_left[valid_indices_pos])

    valid_indices_pos = np.logical_and((tis + 1) < nr_temporal_bins, pos_events_indices)
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)
    index_pos = (xs[valid_indices_pos].astype(np.int64)) + (ys[valid_indices_pos].astype(np.int64)) * width + \
    (tis[valid_indices_pos].astype(np.int64)) * width * height
    np.add.at(voxel_grid_positive, index_pos, vals_right[valid_indices_pos])
    
    # Negative Voxels Grid
    valid_indices_neg = np.logical_and(tis < nr_temporal_bins, ~pos_events_indices)
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)
    np.add.at(voxel_grid_negative,  (xs[valid_indices_neg].astype(np.int64)) + (ys[valid_indices_neg].astype(np.int64)) * width +
              (tis[valid_indices_neg].astype(np.int64)) * width * height, vals_left[valid_indices_neg])

    valid_indices_neg = np.logical_and((tis + 1) < nr_temporal_bins, ~pos_events_indices)
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)
    np.add.at(voxel_grid_negative,  (xs[valid_indices_neg].astype(np.int64)) + (ys[valid_indices_neg].astype(np.int64)) * width +
              (tis[valid_indices_neg].astype(np.int64)) * width * height, vals_right[valid_indices_neg])
    """
  
    """
    # 在这里使用了int64   Positive Voxels Grid
    valid_indices_pos = np.logical_and(tis < nr_temporal_bins, pos_events_indices)
    valid_pos = (xs < width) & (xs >= 0) & (ys < height) & (ys >= 0) & (ts >= 0) & (ts < nr_temporal_bins)
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)

    np.add.at(voxel_grid_positive, xs[valid_indices_pos] + ys[valid_indices_pos] * width +
              tis[valid_indices_pos] * width * height, vals_left[valid_indices_pos])

    valid_indices_pos = np.logical_and((tis + 1) < nr_temporal_bins, pos_events_indices)
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)
    np.add.at(voxel_grid_positive, xs[valid_indices_pos] + ys[valid_indices_pos] * width +
              (tis[valid_indices_pos] + 1) * width * height, vals_right[valid_indices_pos])

    # Negative Voxels Grid
    valid_indices_neg = np.logical_and(tis < nr_temporal_bins, ~pos_events_indices)
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)

    np.add.at(voxel_grid_negative, xs[valid_indices_neg] + ys[valid_indices_neg] * width +
              tis[valid_indices_neg] * width * height, vals_left[valid_indices_neg])

    valid_indices_neg = np.logical_and((tis + 1) < nr_temporal_bins, ~pos_events_indices)
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)
    np.add.at(voxel_grid_negative, xs[valid_indices_neg] + ys[valid_indices_neg] * width +
              (tis[valid_indices_neg] + 1) * width * height, vals_right[valid_indices_neg])

    """


    # voxel_grid_positive = np.reshape(voxel_grid_positive, (nr_temporal_bins, height, width)) # 原方法
    # voxel_grid_negative = np.reshape(voxel_grid_negative, (nr_temporal_bins, height, width))
    # if separate_pol:
    #     return np.concatenate([voxel_grid_positive, voxel_grid_negative], axis=0)
    # voxel_grid = voxel_grid_positive - voxel_grid_negative      # 单单打印一个positive也是有网格的。
    # voxel_grid = voxel_grid_positive 
    # print(voxel_grid.shape) # (3, 480, 640)

    return voxel_grid_np


if __name__ == "__main__":

    # An event sample for testing
    sample_path = '/home/bochen/Research/Dataset/Object_Classification_DVS/N-Caltech101/training/airplanes/airplanes_0.npy'
    events = np.load(sample_path)
    events[:, 2] = (events[:, 2] - events[0, 2]) * 1e6 # Time begins at 0 us
    print(events.shape)
    sensor_size = (180, 240)
    voxel_grid = generate_voxel_grid(events=events, shape=sensor_size, nr_temporal_bins=3)
    print(voxel_grid.shape)
