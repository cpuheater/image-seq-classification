import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import math
import torch
import torch.nn as nn
from src.model import ImageSeqClassifier
import os
from src.dataset import ImageSeqDataset
from torchmetrics import F1Score


parser = argparse.ArgumentParser(description='train')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=12, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--grad-clip', default=5, help='gradient clipping value')
parser.add_argument('--model-dir', default='trained_models', help='dir saving model')
parser.add_argument('--hidden-size', default=64)
parser.add_argument('--data-dir', default="data", help='')

args = parser.parse_args(args=[])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_name = f'{os.path.basename(__file__).rstrip(".py")}_{datetime.now().strftime("%m-%d-%Y_%H:%M:%S")}'
writer = SummaryWriter(f"runs/{exp_name}")

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def accuracy(pred, target):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == target.squeeze()).item()

def save_model(model):
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    torch.save(model.state_dict(),
               os.path.join(args.model_dir, f'{exp_name}.pt'))

def calc_f1(pred, target):
    f1 = F1Score(1).to(device)
    pred = torch.round(pred.squeeze())
    return f1(pred, target)

def train(epoch, train_loader, valid_loader, criterion, start):
    train_loss = 0
    train_accuracy = 0
    model.train()
    for i, seq in enumerate(train_loader, 1):
        (image, seq_length), target = seq
        image, target = image.to(device), target.to(device)
        pred = model(image, seq_length)
        loss = criterion(pred, target.unsqueeze(1).float())
        train_loss += loss.data.item()
        model.zero_grad()
        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        train_accuracy += accuracy(pred, target)

    print('[{}] Train Epoch: {} Loss: {:.2f} Accuracy: {:.6f}'.format(time_since(start),
                                                                      epoch, train_loss, 100 * train_accuracy/len(train_loader.dataset)))
    writer.add_scalar("train/loss", train_loss, epoch)
    writer.add_scalar("train/accuracy", 100 * train_accuracy/len(train_loader.dataset), epoch)
    valid_loss = 0
    valid_accuracy = 0
    valid_epoch_pred = []
    valid_epoch_target = []
    model.eval()
    with torch.no_grad():
        for i, seq in enumerate(valid_loader, 1):
            (image, seq_length), target = seq
            image, target = image.to(device), target.to(device)
            pred = model(image, seq_length)
            loss = criterion(pred, target.unsqueeze(1).float())
            valid_loss += loss.data.item()
            valid_accuracy += accuracy(pred, target)
            valid_epoch_pred.append(pred)
            valid_epoch_target.append(target)

    valid_f1 = calc_f1(torch.cat(valid_epoch_pred), torch.cat(valid_epoch_target))
    writer.add_scalar("valid/loss", valid_loss, epoch)
    writer.add_scalar("valid/accuracy", 100 * valid_accuracy/len(valid_loader.dataset), epoch)
    writer.add_scalar("valid/F1", valid_f1, epoch)
    print('[{}] Valid Epoch: {} Loss: {:.6f} Accuracy: {:.2f} F1: {:.6f}'.format(time_since(start), epoch, valid_loss,
                                                                      100 * valid_accuracy/len(valid_loader.dataset), valid_f1))
    return


if __name__ == '__main__':
    train_dataset = ImageSeqDataset(f"{args.data_dir}/train", transform=None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = ImageSeqDataset(f"{args.data_dir}/valid", transform=None)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    model = ImageSeqClassifier(channels=1, hidden_size=args.hidden_size, output_size=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    start = time.time()
    print("Training for %d epochs..." % args.epochs)
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader, valid_loader, criterion, start)

    save_model(model)

