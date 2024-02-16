import torch
import torch.utils as utils
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

class ImageDataset(utils.data.Dataset):

    def __init__(self, transforms, is_val=False, val_stride=5):
        imgs = []
        self.transform = transforms
        with open('dataset/label.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('\n')[0]
                words = line.split('\t')
                imgs.append((words[0], words[1]))

        val_set = imgs[::val_stride]
        del imgs[::val_stride]
        train_set = imgs

        if is_val:
            self.imgs = val_set
        else:
            self.imgs = train_set

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        #img = transforms.ToTensor()(img)
        label = torch.tensor([float(label)])
        return img, label

    def __iter__(self):
        return iter(self.imgs)

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 将图像的长和宽都调整为256像素
        transforms.ToTensor()  # 将图像转换为torch张量
    ])
    worm_image_dataset = ImageDataset(transforms=transform)
    pic, label = worm_image_dataset.__getitem__(0)
    print(pic.shape)
    print(pic)
    print(label)
    plt.imshow(pic.permute(1, 2, 0))
    plt.show()