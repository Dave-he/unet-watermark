import os
import shutil
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from transformers import AutoImageProcessor, AutoModel
import torch

# 1. 初始化模型
model_name = "facebook/dinov2-large"  # 可选：small/base/large
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda")

# 2. 特征提取函数
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]  # 全局平均池化

# 3. 遍历文件夹提取特征
image_dir = "data/train/watermarked"
features = []
image_paths = []
for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    feat = extract_features(img_path)
    features.append(feat)
    image_paths.append(img_path)
features = np.array(features)

# 4. 聚类分组（自动确定类别数）
cluster = DBSCAN(eps=0.5, min_samples=3)  # 调整eps控制相似度阈值
labels = cluster.fit_predict(features)

# 5. 创建分类文件夹并复制图片
output_dir = "/data/classfy"
os.makedirs(output_dir, exist_ok=True)
for path, label in zip(image_paths, labels):
    class_dir = os.path.join(output_dir, f"class_{label}")
    os.makedirs(class_dir, exist_ok=True)
    shutil.copy(path, class_dir)