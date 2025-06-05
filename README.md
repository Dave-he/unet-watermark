# unet检测水印

## 0. 项目介绍
这是一个使用U-Net模型进行水印检测的项目。项目使用了PyTorch和Segmentation Models PyTorch（SMP）库来构建和训练模型。
## 安装依赖
```bash
#venv
virtualenv env -p 3.13
source env/bin/activate #linux
./env/scripts/activate.ps1 #windows 

#conda
conda create -n env313 python=3.13

pip install -r requirements.txt
```

# 项目结构
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

#复制到train目录下
find /Users/hyx/unet-watermark/data/WatermarkDataset/images/val/ -type f -exec cp {} /Users/hyx/unet-watermark/data/train/watermarked/ \;

#根据label框生成mask图片(从yolo格式转换为mask图片)
python src/scripts/enhance_masks.py
#修复图片
python src/scripts/image_fixer.py data/train
```


## 2. 训练 && 预测
```bash
 1. 训练模型（使用GPU，300轮，每50轮保存检查点）
python main.py train --device cuda --epochs 300 --batch-size 16

python main.py train \
    --config src/configs/unet_watermark.yaml \
    --device cuda \
    --epochs 300 \
    --batch-size 16 \
    --output-dir logs/unet_experiment \
    --model-save-path models/unet_watermark_best.pth


# 使用不同的检查点进行预测比较
python main.py predict --input test.jpg --output results1 --model models/checkpoints/checkpoint_epoch_050.pth
python main.py predict --input test.jpg --output results2 --model models/checkpoints/checkpoint_epoch_100.pth
python main.py predict --input test.jpg --output results3 --model models/checkpoints/checkpoint_epoch_150.pth
```




# 上传数据集

huggingface-cli upload heyongxian/watermark_images ./data --repo-type=dataset


```
7z a -v500m data.7z ./data

-mx9 最高压缩级别，-mmt=8 使用 8 线程加速
7z a -t7z -mx9 -mmt=8 final.7z ./

aws s3 --no-sign-request \
  --endpoint http://bs3-sgp.internal \
  cp mark-pro.zip s3://oversea-game/ai-model/

```

国内加速
HF_ENDPOINT=https://hf-mirror.com

#下载
https://ghproxy.net/
https://github.moeyy.xyz/