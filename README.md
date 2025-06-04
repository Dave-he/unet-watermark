# unet检测水印

#项目结构
```bash
/Users/hyx/unet-watermark/
├── main.py                    # 简洁的入口文件
├── requirements.txt           # 依赖文件
├── README.md
├── data/                      # 数据目录
└── src/                       # 所有代码逻辑
    ├── cli.py                 # 命令行接口逻辑
    ├── train_smp.py           # 训练逻辑
    ├── predict_smp.py         # 预测逻辑
    ├── configs/               # 配置文件
    │   ├── config.py
    │   └── unet_watermark.yaml
    ├── models/                # 模型定义
    │   └── smp_models.py
    └── utils/                 # 工具模块
        ├── dataset.py
        ├── losses.py
        └── metrics.py
```


## 1. 数据准备

https://kaggle.com/
配置 API 密钥
登录 Kaggle 账户，进入账户设置（Account Settings）。
点击 “Create New API Token”，下载kaggle.json文件。
将文件保存到默认路径：
Linux/macOS：~/.kaggle/kaggle.json
Windows：C:\Users\<用户名>\.kaggle\kaggle.json

```bash
pip install kaggle
#!/bin/bash
kaggle datasets download -d kamino/largescale-common-watermark-dataset -p data/
kaggle datasets download -d felicepollano/watermarked-not-watermarked-images -p data/
```