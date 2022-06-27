
import random
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import uuid
import os
import torch
import argparse

parser = argparse.ArgumentParser(description='create dataset')
parser.add_argument('--data-dir', default="data", help='')

args = parser.parse_args(args=[])

def gen_normal_int():
    while True:
        num = round(random.gauss(10, 3))
        if 3 <= num <= 30:
            yield num


def generate_sequences(total):
    sequences = []
    count = 0
    for n in gen_normal_int():
        if count + n > total:
            print(f"Total examples:  {count}")
            break
        sequences.append(n)
        count += n
    return sequences


def create_dataset(dst_path, is_train=True):

    dst_path = os.path.join(dst_path, "train") if is_train else os.path.join(dst_path, "valid")
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    dataset = torchvision.datasets.MNIST(".", download=True, train=is_train,
                                         transform=transforms.Compose([torchvision.transforms.ToTensor()]))
    loader = DataLoader(dataset, batch_size=1, drop_last=True, num_workers=0)
    train_loader_tier = iter(loader)
    count_pos = 0
    count_neg = 0
    for n in generate_sequences(len(loader)):
        targets = []
        data_point = torch.zeros(n, 28, 28)
        for i, e in enumerate(range(n)):
            img, target = next(train_loader_tier)
            data_point[i] = img[0, 0]
            targets.append(target.item())
        file_name = os.path.join(dst_path, f"{str(uuid.uuid4())}.npy")
        target = 1 if any(e == 4 for e in targets) else 0
        if target == 1:
            count_pos += 1
        else:
            count_neg += 1
        data_point = [data_point, target]
        torch.save(data_point, file_name)
    print(f"Total number of newly created examples: {count_pos + count_neg}, positive: {count_pos}, negative:  ${count_neg}")

if __name__ == '__main__':
    create_dataset(args.data_dir, True)
    create_dataset(args.data_dir, False)


