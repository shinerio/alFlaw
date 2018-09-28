from torch.utils.data import Dataset
from PIL import Image


# 测试集图片读取
class TestDataset(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row['img_path']))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)
