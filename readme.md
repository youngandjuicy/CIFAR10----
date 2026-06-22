# 项目结构

cifar10_project/
│
├── configs/
│   └── config.py
│
├── datasets/
│   └── cifar10.py
│
├── models/
│   └── tudui.py
│
├── engine/
│   ├── train.py
│   └── evaluate.py
│
├── utils/
│   ├── seed.py
│   ├── checkpoint.py
│   └── logger.py
│
├── checkpoints/
│
├── runs/
│
├── train.py
│
└── requirements.txt



- 从A文件中python文件中import某个类到B文件，这个类的构建需要用到某些库，那么A文件中import的库还需要重新在B文件中引入吗？
- 训练轮数太多会导致模型过拟合吗？还是说过拟合是模型本身结构设计的硬伤，比如参数过多，而不是训练过程的影响？
- 定义测试函数前@torch.no_grad()作用是什么，需要import torch吗
- setseed怎么用
- checkpoint的保存路径path怎么写
