import numpy as np
import torch


def std_mean(mydataset):
    print('开始计算cal，mean')
    dataset_loader = torch.utils.data.DataLoader(dataset=mydataset,
                                                 batch_size=20000,
                                                 shuffle=True, num_workers=0)
    data = iter(dataset_loader).__next__()[0]
    # Here axis is a tuple (0, 2, 3), which means that the numbers along the 0th, 2nd, and 3rd dimensions
    # are summed and averaged to find the value of dimension 1
    data_mean = np.mean(data.numpy(), axis=(0, 2, 3))
    # Assuming the dimension of the train is [128,3,64,64], then it's averaging over 128*64*64 numbers.
    data_std = np.std(data.numpy(), axis=(0, 2, 3))
    # print("三通道的mean值：{}，std: {}".format(data_mean, data_std))
    return data_mean, data_std
