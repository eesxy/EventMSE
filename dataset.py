from torch.utils.data import Dataset
import os
import h5py
import numpy as np

# dataloader
# 与matlab代码配合使用
class EventData(Dataset):
    def __init__(self, inputpath):
        self.inputPath = inputpath
        self.nameList = os.listdir(inputpath)

    def __getitem__(self, index):
        inputpath = os.path.join(self.inputPath, self.nameList[index])
        with h5py.File(inputpath, 'r') as data:
            ret_input = data['data']['input'][()].T
            ret_target = data['data']['target'][()].T
        ret_input = np.expand_dims(ret_input, axis=0)
        ret_target = np.expand_dims(ret_target, axis=0)
        return ret_input, ret_target

    def __len__(self):
        return len(self.nameList)
