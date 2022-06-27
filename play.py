
import argparse
import torch
from src.model import ImageSeqClassifier
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

parser = argparse.ArgumentParser(description='play')
parser.add_argument('--model-path', default='trained_models/train_06-27-2022_12:14:21.pt', help='dir with the model')
parser.add_argument('--hidden-size', default=64)
args = parser.parse_args(args=[])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    model = ImageSeqClassifier(channels=1, hidden_size=args.hidden_size, output_size=1).to(device).eval()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    dataset = torchvision.datasets.MNIST(".", download=True, train=False,
                                         transform=transforms.Compose([torchvision.transforms.ToTensor()]))
    loader = DataLoader(dataset, batch_size=1, drop_last=True, num_workers=0)
    train_loader_tier = iter(loader)
    while True:
        n = random.randint(1, 400)
        targets = []
        data_point = torch.zeros(n, 28, 28).to(device)
        try:
            for i, e in enumerate(range(n)):
                img, target = next(train_loader_tier)
                img = img.to(device)
                data_point[i] = img[0, 0]
                targets.append(target.item())
            target = 1 if any(t == 4 for t in targets) else 0
        except StopIteration:
            break
        with torch.no_grad():
            pred = model(data_point.unsqueeze(0), torch.LongTensor([n]))
        if target == int(torch.round(pred.squeeze()).item()):
            print("Correct")
        else:
            print("Incorrect !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")



