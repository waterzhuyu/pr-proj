import os
import time
import argparse

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pretrain
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch CIFAR10 Training')
    parser.add_argument('-lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--loss', default='ce', help='Loss function')
    parser.add_argument('--opt', default='adam', help='Optimizer')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--bs', default='128', help='Batch Size')
    parser.add_argument('--size', default=384, type=int, help='Image Size')
    parser.add_argument('--n_epochs', default='20', help='Total number of training rounds')
    parser.add_argument('--pretrain', action='store_true', help='Use model pretrained on ImageNet-21k.')
    parser.add_argument('--finetune_all', action='store_true',
                        help='update the weights of the whole model when running finetune')

    args = parser.parse_args()

    size = args.size
    n_epochs = args.n_epochs
    lr = args.lr
    pretrained = args.pretrain
    finetune_all = args.finetune_all

    # Data Augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    model = pretrain.ViT(name='B_16', num_classes=10, image_size=size, pretrained=pretrained)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = nn.DataParallel(model)

    # Dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=6)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=6)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)

    # auto mix precision
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    watermark = "{}_lr{}".format("ViT", lr)
    wandb.init(project="ViT-finetune", name=watermark)
    wandb.config.update(args)

    def train(current_epoch):
        print(f'epoch: {current_epoch}\n')
        model.train()
        train_loss = 0
        num = 0
        correct = 0
        tk = tqdm(trainloader, desc="EPOCH" + "[TRAIN]" + str(current_epoch+1) + '/' + str(200))
        for batch_index, (inputs, targets) in enumerate(tk):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            num += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            current_loss = train_loss / (batch_index + 1)
            accuracy = 100. * correct / num
            tk.set_postfix({"LOSS": "%6f" % float(current_loss), "ACC": "%6f" % float(accuracy)})

        return current_loss

    best_accuracy = 0

    def val(epoch):
        global best_accuracy
        model.eval()
        val_loss = 0
        num = 0
        correct = 0
        all_pred = []
        all_targets = []
        tk = tqdm(testloader, desc="EPOCH" + "[TEST]" + str(epoch + 1) + '/' + str(200))
        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(tk):
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)

                all_pred.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                num += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                current_loss = val_loss / (batch_index + 1)
                accuracy = 100. * correct / num
                tk.set_postfix({"LOSS": "%6f" % float(current_loss), "ACC": "%6f" % float(accuracy)})

            class_accuracy = classification_report(all_targets, all_pred, target_names=classes)

            if not os.path.isdir('result'):
                os.mkdir('result')

            output_file_path = f'./result/per_class_accuracy_{args.net}_epoch{epoch}.txt'
            with open(output_file_path, 'w') as output_file:
                output_file.write("Per-class Accuracy:\n" + class_accuracy)

            # Generate and save confusion matrix
            cm = confusion_matrix(all_targets, all_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.savefig(f'./result/confusion_matrix_{args.net}_epoch{epoch}.png')

            # save checkpoint
            if accuracy > best_accuracy:
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict()
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, 'checkpoint/'+f'B_16.ckpt.t7')

            os.makedirs('log', exist_ok=True)
            content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, ' \
                                           f'val loss: {val_loss:.5f}, acc: {accuracy :.5f}'
            print(content)
            with open(f'log/logB_16.txt', 'a') as f:
                f.write(content + '\n')
            return val_loss, accuracy

    lss = []
    acc = []

    model.cuda()
    for n_epoch in range(n_epochs):
        start = time.time()
        train_lss = train(n_epoch)
        val_lss, accu = val(n_epoch)

        lr_scheduler.step()

        lss.append(val_lss)
        acc.append(accu)

        wandb.log({'epoch': n_epoch, 'train_loss': train_lss, 'val_loss': val_lss,
                   'val_acc': accu, 'lr': optimizer.param_groups[0]['lr'], 'epoch_time': time.time()-start})

    flag = 1 if finetune_all else 2
    wandb.save("wandb_finetune_strategy{}.h5".format(flag))
