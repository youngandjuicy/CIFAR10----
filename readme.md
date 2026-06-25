# CIFAR-10 Image Classification

一个基于 PyTorch 的 CIFAR-10 图像分类项目。项目包含数据加载、模型定义、训练、评估、随机种子固定、checkpoint 保存与恢复训练等完整流程，适合作为深度学习图像分类任务的基础模板。

## 项目特点

- 使用 `torchvision.datasets.CIFAR10` 自动下载和加载数据集
- 支持训练集数据增强：随机裁剪、随机水平翻转、归一化
- 支持两种模型：
  - `tudui`：简单 CNN 分类模型
  - `resnet18`：适配 CIFAR-10 输入尺寸的 ResNet-18
- 支持保存最佳模型 `best.pth`
- 支持保存最新训练状态 `latest.pth`
- 支持从 checkpoint 恢复训练
- 使用配置文件统一管理超参数

## 项目结构

```text
CIFAR10分类任务/
├── cifar10_main.py        # 训练入口
├── configs/
│   └── config.py          # 超参数与路径配置
├── data/
│   └── cifar10.py         # CIFAR-10 数据加载与预处理
├── engine/
│   ├── train.py           # 单轮训练逻辑
│   └── evaluate.py        # 测试集评估逻辑
├── models/
│   ├── tudui.py           # 简单 CNN 模型
│   └── resnet.py          # ResNet-18 模型
├── utils/
│   ├── checkpoint.py      # checkpoint 保存与加载
│   └── seed.py            # 随机种子设置
└── readme.md
```

## 环境要求

建议使用 Python 3.8 或更高版本。

主要依赖：

```bash
pip install torch torchvision
```

如果需要安装 GPU 版本的 PyTorch，请根据自己的 CUDA 版本参考 PyTorch 官网安装命令：

https://pytorch.org/get-started/locally/

## 配置说明

主要配置位于 `configs/config.py`：

```python
class Config:
    data_root = "./dataset"
    checkpoint_dir = "./checkpoints"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = "resnet18"  # 可选: "tudui", "resnet18"

    batch_size = 64
    num_epochs = 100
    num_classes = 10

    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 5e-4

    save_best = True
    save_latest = True
    resume = False
    resume_path = "./checkpoints/latest.pth"
```

切换模型时，修改：

```python
model = "resnet18"
```

或：

```python
model = "tudui"
```

## 开始训练

在项目根目录运行：

```bash
python cifar10_main.py
```

第一次运行时，程序会自动下载 CIFAR-10 数据集到 `dataset/` 目录。

训练过程中会输出每个 epoch 的训练损失、训练准确率、测试损失和测试准确率，例如：

```text
Epoch [1/100] - Train Loss: 1.2345, Train Acc: 55.20%, Test Loss: 1.0123, Test Acc: 64.80%
```

## 恢复训练

默认会在 `checkpoints/` 目录下保存：

- `best.pth`：测试集准确率最高的模型
- `latest.pth`：最近一次 epoch 结束后的完整训练状态

从默认 checkpoint 恢复训练：

```bash
python cifar10_main.py --resume
```

从指定 checkpoint 恢复训练：

```bash
python cifar10_main.py --resume checkpoints/best.pth
```

也可以在 `configs/config.py` 中启用默认恢复：

```python
resume = True
resume_path = "./checkpoints/latest.pth"
```

## Checkpoint 内容

checkpoint 中保存了以下内容：

- 当前训练轮数 `epoch`
- 模型参数 `model`
- 优化器状态 `optimizer`
- 学习率调度器状态 `scheduler`
- 当前最佳准确率 `best_acc`

因此恢复训练后，优化器动量、学习率调度进度和最佳准确率都会一起恢复。

## GitHub 上传说明

本项目已经通过 `.gitignore` 忽略以下本地生成内容：

- `dataset/`
- `checkpoints/`
- `__pycache__/`
- `.vscode/`
- 日志和临时文件

这些文件通常不需要上传到 GitHub。上传源码即可，其他用户运行训练脚本时会自动下载数据集并生成 checkpoint。

## 后续改进方向

- 增加命令行参数配置模型、学习率、batch size 等超参数
- 增加 TensorBoard 或 CSV 日志记录
- 增加更多模型结构，例如 VGG、MobileNet、WideResNet
- 增加测试脚本，用于加载 `best.pth` 并单独评估模型
- 增加训练曲线可视化
