import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from PIL import Image
import logging
import warnings
import contextlib
import sys
import os

# 添加日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WatermarkDataset(Dataset):
    """水印数据集类"""
    
    def __init__(self, watermarked_dirs, clean_dirs=None, mask_dirs=None,
                 transform=None, mode='train', generate_mask_threshold=30, 
                 cache_images=True, prefetch_factor=2):
        """
        初始化数据集
        
        Args:
            watermarked_dirs (list): 带水印图像目录列表
            clean_dirs (list): 干净图像目录列表
            mask_dirs (list): 掩码目录列表
            transform: 数据增强变换
            mode (str): 模式 ('train', 'val', 'test')
            generate_mask_threshold (int): 生成掩码的阈值
            cache_images (bool): 是否缓存图像
            prefetch_factor (int): 预取因子
        """
        self.watermarked_dirs = watermarked_dirs if isinstance(watermarked_dirs, list) else [watermarked_dirs]
        self.clean_dirs = clean_dirs if isinstance(clean_dirs, list) else [clean_dirs] if clean_dirs else []
        self.mask_dirs = mask_dirs if isinstance(mask_dirs, list) else [mask_dirs] if mask_dirs else []
        self.transform = transform
        self.mode = mode
        self.generate_mask_threshold = generate_mask_threshold
        self.cache_images = cache_images
        self.prefetch_factor = prefetch_factor
        self.image_cache = {} if cache_images else None
        
        # 初始化图像文件列表
        self.image_files = self._collect_image_files()
        
        # 预加载小数据集到内存
        if cache_images and len(self.image_files) < 1000:
            self._preload_images()
    
    def _collect_image_files(self):
        """收集所有图像文件路径"""
        image_files = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for watermarked_dir in self.watermarked_dirs:
            if os.path.exists(watermarked_dir):
                for filename in os.listdir(watermarked_dir):
                    if any(filename.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(watermarked_dir, filename))
        
        logger.info(f"找到 {len(image_files)} 个图像文件")
        return sorted(image_files)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_files)
    
    def _preload_images(self):
        """预加载图像到内存缓存"""
        logger.info("预加载图像到内存...")
        for idx, image_path in enumerate(tqdm(self.image_files[:100])):
            try:
                img = self._safe_imread(image_path)
                if img is not None:
                    self.image_cache[idx] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.warning(f"预加载失败: {image_path}, {e}")
    
    def __getitem__(self, idx):
        # 使用缓存的图像
        if self.image_cache and idx in self.image_cache:
            watermarked_img = self.image_cache[idx]
            image_name = os.path.basename(self.image_files[idx])
        else:
            watermarked_path = self.image_files[idx]
            image_name = os.path.basename(watermarked_path)
        
        # 读取带水印图像 - 添加错误处理
        watermarked_img = self._safe_imread(watermarked_path)
        if watermarked_img is None:
            # 如果图片损坏，尝试使用下一张图片
            logger.warning(f"跳过损坏的图片: {watermarked_path}")
            return self.__getitem__((idx + 1) % len(self.image_files))
        
        watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB)
        
        # 获取或生成掩码
        mask = self._get_or_generate_mask(image_name, watermarked_img)
        
        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=watermarked_img, mask=mask)
            watermarked_img = augmented['image']
            mask = augmented['mask']
        
        # 转换为张量
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0) if len(mask.shape) == 2 else mask
        mask = mask / 255.0 if mask.max() > 1.0 else mask
        # 确保掩码值在 0-1 范围内，然后转换为整数类型
        mask = torch.clamp(mask, 0.0, 1.0)
        # 二值化掩码：>0.5 为 1，否则为 0，并转换为 long 类型
        mask = (mask > 0.5).long()
        # 移除多余的维度，确保形状为 (H, W)
        mask = mask.squeeze()
        
        return watermarked_img, mask
    
    @contextlib.contextmanager
    def suppress_opencv_warnings():
        """抑制OpenCV警告"""
        old_stderr = sys.stderr
        try:
            with open(os.devnull, 'w') as devnull:
                sys.stderr = devnull
                yield
        finally:
            sys.stderr = old_stderr
    
    def _safe_imread(self, image_path, max_retries=2):  # 减少重试次数
        """优化的安全图像读取"""
        for attempt in range(max_retries):
            try:
                # 直接使用OpenCV读取，减少PIL验证步骤
                with suppress_opencv_warnings():
                    img = cv2.imread(image_path)
                    if img is not None and img.size > 0:
                        return img
                        
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(f"读取图片失败: {image_path}, 错误: {str(e)}")
                    return None
                
            return None
    
    def _get_or_generate_mask(self, image_name, watermarked_img):
        """获取或生成掩码"""
        # 首先尝试从mask_dirs加载现有掩码
        for m_dir in self.mask_dirs:
            mask_name = os.path.splitext(image_name)[0] + '.png'
            mask_path = os.path.join(m_dir, mask_name)
            
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    return mask
        
        # 如果没有现有掩码，尝试生成掩码
        for c_dir in self.clean_dirs:
            clean_path = os.path.join(c_dir, image_name)
            if os.path.exists(clean_path):
                clean_img = cv2.imread(clean_path)
                if clean_img is not None:
                    clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
                    mask = self._generate_mask(watermarked_img, clean_img)
                    
                    # 保存生成的掩码到第一个mask_dir
                    if self.mask_dirs:
                        mask_name = os.path.splitext(image_name)[0] + '.png'
                        mask_path = os.path.join(self.mask_dirs[0], mask_name)
                        cv2.imwrite(mask_path, mask)
                    
                    return mask
        
        # 如果无法获取掩码，返回全零掩码
        return np.zeros(watermarked_img.shape[:2], dtype=np.uint8)
    
    def _generate_mask(self, watermarked_img, clean_img):
        """通过比较带水印图像和干净图像生成掩码"""
        # 确保图像尺寸一致
        if watermarked_img.shape != clean_img.shape:
            clean_img = cv2.resize(clean_img, (watermarked_img.shape[1], watermarked_img.shape[0]))
        
        # 计算差异
        diff = cv2.absdiff(watermarked_img, clean_img)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        
        # 二值化生成掩码
        _, mask = cv2.threshold(diff_gray, self.generate_mask_threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return mask

def get_train_transform(img_size=512):
    """获取训练时的数据增强变换"""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transform(img_size=512):
    """获取验证时的数据变换"""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def create_datasets(cfg):
    """创建训练集和验证集"""
    # 创建完整数据集
    # 准备多个数据目录
    watermarked_dirs = [os.path.join(cfg.DATA.ROOT_DIR, "watermarked")]
    clean_dirs = [os.path.join(cfg.DATA.ROOT_DIR, "clean")]
    mask_dirs = [os.path.join(cfg.DATA.ROOT_DIR, "masks")]

    if cfg.DATA.ADDITIONAL_ROOT_DIRS:
        for additional_root in cfg.DATA.ADDITIONAL_ROOT_DIRS:
            watermarked_dirs.append(os.path.join(additional_root, "watermarked"))
            clean_dirs.append(os.path.join(additional_root, "clean"))
            mask_dirs.append(os.path.join(additional_root, "masks"))

    full_dataset = WatermarkDataset(
        watermarked_dirs=watermarked_dirs,
        clean_dirs=clean_dirs,
        mask_dirs=mask_dirs,
        transform=get_train_transform(cfg.DATA.IMG_SIZE),
        mode='train',
        generate_mask_threshold=cfg.DATA.GENERATE_MASK_THRESHOLD
    )
    
    # 设置随机种子确保可复现
    random.seed(cfg.DATA.SEED)
    torch.manual_seed(cfg.DATA.SEED)
    
    # 计算数据集划分
    dataset_size = len(full_dataset)
    train_size = int(cfg.DATA.TRAIN_RATIO * dataset_size)
    val_size = dataset_size - train_size
    
    # 随机划分数据集
    indices = list(range(dataset_size))
    if cfg.DATA.SHUFFLE:
        random.shuffle(indices)
    
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    
    # 创建训练集和验证集
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # 为验证集设置验证变换
    val_dataset.dataset.transform = get_val_transform(cfg.DATA.IMG_SIZE)
    
    print(f"数据集划分完成:")
    print(f"训练集: {len(train_dataset)} 张图像")
    print(f"验证集: {len(val_dataset)} 张图像")
    
    return train_dataset, val_dataset