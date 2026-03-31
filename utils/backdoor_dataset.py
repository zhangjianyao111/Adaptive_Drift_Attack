import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms

class BackdoorDataset(Dataset):
    def __init__(self, base_dataset, target_label=0, poison_ratio=0.1, trigger_size=3):

        self.base_dataset = base_dataset
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.trigger_size = trigger_size

        self.indices = np.arange(len(base_dataset))
        self.num_poison = int(len(base_dataset) * poison_ratio)
        # 这里简单选前 num_poison 个索引，你也可以随机打乱
        self.poison_indices = set(self.indices[:self.num_poison])
        
        # MNIST 的 Normalize
        self.normalize = transforms.Normalize((0.1307,), (0.3081,))

    def _add_trigger(self, img):
        img = img.clone()
        c, h, w = img.shape
        s = self.trigger_size
        img[:, h - s:h, w - s:w] = 1.0
        return img
    
    def __getitem__(self, idx):

        img, label = self.base_dataset[idx]

        # 如果是投毒样本 → 先反归一化 → 加触发器 → 再归一化
        if idx in self.poison_indices:
            # 反归一化
            img = img * 0.5 + 0.5   # 从 [-1,1] 回到 [0,1]

            # 加触发器
            img = self._add_trigger(img)

            # 再归一化
            img = (img - 0.5) / 0.5

            label = self.target_label

        return img, label

    def __len__(self):
        return len(self.base_dataset)
