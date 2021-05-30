import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, models
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F

# 数据增强
img_transforms = {
    'train': transforms.Compose([
        # transforms.RandomSizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(30),
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.2),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.RandomSizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        # transforms.RandomSizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
label_transforms = {
    'train': transforms.Compose([
        # transforms.RandomSizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(30),
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.2),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        # transforms.Resize(224),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        # transforms.RandomSizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.Resize(224),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        # transforms.RandomSizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.Resize(224),
        transforms.ToTensor()
    ])
}


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class MyDataset(Dataset):
    def __init__(self, type, img_path, label_path, txt_path, img_transform=None, label_transform=None):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.idx_list = [line.strip() for line in lines]
            self.img_list = [os.path.join(img_path, line.strip() + '.jpg') for line in lines]
            self.label_list = [os.path.join(label_path, line.strip() + '.png') for line in lines]
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.type = type

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, item):
        img_name = self.img_list[item]
        label_name = self.label_list[item]
        img = Image.open(img_name)
        label = Image.open(label_name)
        if self.type == 'train':
            img = img.resize((520, 520))
            label = label.resize((520, 520))
        if self.img_transform is not None:
            img = self.img_transform(img)
        label = torch.tensor(np.array(label))
        return img, label


class TestDataset(Dataset):
    def __init__(self, img_path, txt_path, img_transform=None, label_transform=None):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.idx_list = [line.strip() for line in lines]
            self.img_list = [os.path.join(img_path, line.strip() + '.jpg') for line in lines]
        self.img_transform = img_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, item):
        idx = self.idx_list[item]
        img_name = self.img_list[item]
        img = Image.open(img_name)
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, idx


img = Image.open('dataset/annotations_trainval/1635.png')  # P
palette = np.array(img.getpalette(), dtype=np.uint8).reshape((256, 3))

if __name__ == '__main__':
    # environment
    use_gpu = torch.cuda.is_available()
    # use_gpu = False
    f = open('record.txt', 'w')
    # parameters
    MAX_EPOCH = 20
    NUM_CLASSES = 21
    INPUTS_NUM = {'train': 1600, 'val': 500}
    BATCH_SIZE = {'train': 2, 'val': 1, 'test': 1}
    # load train & val data
    image_datasets = {
        'train': MyDataset(type='train', img_path='dataset/images', label_path='dataset/annotations_trainval',
                           txt_path='train.txt', img_transform=img_transforms['train'],
                           label_transform=label_transforms['train']),
        'val': MyDataset(type='val', img_path='dataset/images', label_path='dataset/annotations_trainval',
                         txt_path='val.txt', img_transform=img_transforms['val'],
                         label_transform=label_transforms['val']),
        'test': TestDataset(img_path='dataset/images',
                            txt_path='test.txt', img_transform=img_transforms['test'],
                            label_transform=label_transforms['test']),
    }
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE[x], shuffle=True) for x in
                   ['train', 'val', 'test']}
    # model
    # model = models.segmentation.fcn_resnet50(pretrained=True)
    # model = models.segmentation.fcn_resnet101(pretrained=True, num_classes=NUM_CLASSES)
    # model = models.segmentation.deeplabv3_resnet50(pretrained=True, num_classes=NUM_CLASSES)
    model = models.segmentation.deeplabv3_resnet101(pretrained=True, num_classes=NUM_CLASSES)
    # model.classifier = DeepLabHead(2048, NUM_CLASSES)
    # model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, num_classes=NUM_CLASSES)
    # model = models.segmentation.lraspp_mobilenet_v3_large(pretrained=True, num_classes=NUM_CLASSES)
    if use_gpu:
        model = model.cuda()
    # criterion = torch.nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(ignore_index=255, size_average=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # train & validate
    last_val_loss = float('inf')
    last_model_wts = model.state_dict()
    stop = False
    for e in range(MAX_EPOCH):
        for phase in ['val', 'train']:
            if phase == 'train':
                # scheduler.step()  # 更新学习率
                model.train()
            else:
                model.eval()
            tmp_loss = 0
            for data in dataloaders[phase]:
                # load data
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()  # 将网络中的所有梯度置0
                # forward progressing
                outputs = model(inputs)['out']
                loss = criterion(outputs, labels.long())
                # backward progressing
                if phase == 'train':
                    loss.backward()  # 回传损失
                    optimizer.step()  # 更新参数
                # calculate loss and accuracy
                tmp_loss += loss.item()
                # break###############
            tmp_loss /= INPUTS_NUM[phase]
            print('{} Loss: {:.3f}'.format(phase, tmp_loss))
            f.write('{} Loss: {:.3f}\n'.format(phase, tmp_loss))
            # record model
            if phase == 'val':
                # 早停：验证集准确率下降时停止
                if tmp_loss >= last_val_loss:
                    stop = True
                else:
                    last_model_wts = model.state_dict()
                    last_val_loss = tmp_loss
            # break###############
        if stop:
            # store the model
            torch.save(last_model_wts, 'model.pth')
            f.write('Final loss: {:.3f}\n'.format(last_val_loss))
            # generate val images
            model.eval()
            optimizer.zero_grad()
            for data in dataloaders['test']:
                # load data
                inputs, idx = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                else:
                    inputs = Variable(inputs)
                outputs = model(inputs)['out']
                om = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
                img = Image.fromarray(om.astype(np.uint8))
                img.putpalette(palette)
                img.save('test_pred/' + idx[0] + '.png')
            break
    f.close()
