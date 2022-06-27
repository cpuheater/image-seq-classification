from torch.utils.data import Dataset
import os
import torch
from typing import Tuple

class ImageSeqDataset(Dataset):
    def __init__(self, image_dir, transform=None, max_num= 30):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.max_num = max_num

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: torch.Tensor) -> Tuple[Tuple[torch.Tensor, int], torch.Tensor]:
        img_path = os.path.join(self.image_dir, self.images[index])
        image, target = torch.load(img_path)
        if self.transform is not None:
            image = self.transform(image = image)["image"]
        padded_image = torch.zeros(self.max_num, image.size(-1), image.size(-1))
        seq_length = image.size(0)
        padded_image[:seq_length] = image
        return (padded_image, seq_length), target

