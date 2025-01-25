import torch
from dataloader import get_dataloader
from models import UNet, NestedUNet, R2U_Net, AttU_Net, R2AttU_Net, U_Net
from trainer import Trainer

# 模型初始化
models_dict = {
    "UNet": UNet,
    "NestedUNet": NestedUNet,
    "R2U_Net": R2U_Net,
    "AttU_Net": AttU_Net,
    "R2AttU_Net": R2AttU_Net,
    "U_Net": U_Net,
}

select_model = "UNet"
model = models_dict[select_model](n_channels=3, n_classes=1)

# 配置参数
data_folder = "./infrared_data"
csv_file = "./infrared_data/group_split.csv"

batch_size = 8
lr = 1e-4
epochs = 200
patience = 10
checkpoint_dir = "./checkpoints"
debug = False

criterion = torch.nn.BCEWithLogitsLoss()
optimizer_class = torch.optim.Adam
scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
scheduler_kwargs = {
    "mode": "min",
    "factor": 0.1,
    "patience": 2,
}

# 获取数据加载器
train_loader = get_dataloader(data_folder, csv_file, "train", batch_size, debug)
val_loader = get_dataloader(data_folder, csv_file, "val", batch_size, debug)
test_loader = get_dataloader(data_folder, csv_file, "test", batch_size, debug)

# 设备设置
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=device,
    lr=lr,
    epochs=epochs,
    patience=patience,
    checkpoint_dir=checkpoint_dir,
    criterion=criterion,
    optimizer_class=optimizer_class,
    scheduler_class=scheduler_class,
    scheduler_kwargs=scheduler_kwargs,
)

checkpoint = "./checkpoints/UNet_20250125_175610/best_model.pth"

# trainer.train()
trainer.test(checkpoint)
trainer.visualize_predictions(checkpoint)
