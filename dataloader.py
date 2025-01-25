from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class InfraredData(Dataset):
    def __init__(self, data_folder, csv_file, split, debug=False):
        self.data = pd.read_csv(csv_file)
        self.split = split

        if self.split in ["train", "val", "test"]:
            self.data = self.data[self.data["Split"] == split]
        if debug:
            self.data = self.data.head(10)

        self.infrared_path = [
            Path(data_folder) / Path(x) for x in self.data["Infrared"].values
        ]
        self.mask_path = [Path(data_folder) / Path(x) for x in self.data["Mask"].values]

        self.train_albumentations = A.Compose(
            [
                # A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256), antialias=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256), antialias=True),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 读取 infrared 和 mask
        infrared = cv2.imread(str(self.infrared_path[index]))
        infrared = cv2.cvtColor(infrared, cv2.COLOR_BGR2RGB)  # 转换为 RGB

        mask = cv2.imread(str(self.mask_path[index]), cv2.IMREAD_GRAYSCALE)
        mask = mask / 255 if (mask > 1).any() else mask

        # 保存未增强的原图和掩码
        original_infrared = infrared.copy()
        original_mask = mask.copy()

        # 数据增强
        if self.split == "train":
            augmented = self.train_albumentations(image=infrared, mask=mask)
            infrared = augmented["image"]
            mask = augmented["mask"]

        # 转换为张量
        infrared = self.image_transform(infrared).float()
        mask = self.mask_transform(mask).float()

        # 返回增强后的张量和未增强的原图
        return infrared, mask, original_infrared, original_mask


def get_dataloader(data_folder, csv_file, split, batch_size, debug=False):
    infrared_data = InfraredData(data_folder, csv_file, split, debug)
    infrared_loader = DataLoader(
        infrared_data,
        batch_size=batch_size,
        shuffle=True if split == "train" else False,
    )
    return infrared_loader


if __name__ == "__main__":
    data_folder = "./3-Resized"
    csv_file = "./3-Resized/group_split.csv"
    infrared_train = InfraredData(data_folder, csv_file, "train")
    infrared_val = InfraredData(data_folder, csv_file, "val")
    infrared_test = InfraredData(data_folder, csv_file, "test")

    train_loader = DataLoader(infrared_train, batch_size=4, shuffle=True)
    val_loader = DataLoader(infrared_val, batch_size=4, shuffle=False)
    test_loader = DataLoader(infrared_test, batch_size=4, shuffle=False)

    for i, (infrared, mask) in enumerate(train_loader):
        print(infrared.shape, mask.shape)  # 输出形状
        if i == 0:
            break
