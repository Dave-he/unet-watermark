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
from tqdm import tqdm

# 添加日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WatermarkDataset(Dataset):
    """水印数据集类"""
    
    def __init__(self, watermarked_dirs, clean_dirs=None, mask_dirs=None,
                 transform=None, mode='train', generate_mask_threshold=30, 
                 cache_images=True, prefetch_factor=2, use_blurred_mask=False):
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
            use_blurred_mask (bool): 是否使用模糊的mask进行训练，默认False生成精确mask
        """
        self.watermarked_dirs = watermarked_dirs if isinstance(watermarked_dirs, list) else [watermarked_dirs]
        self.clean_dirs = clean_dirs if isinstance(clean_dirs, list) else [clean_dirs] if clean_dirs else []
        self.mask_dirs = mask_dirs if isinstance(mask_dirs, list) else [mask_dirs] if mask_dirs else []
        self.transform = transform
        self.mode = mode
        self.generate_mask_threshold = generate_mask_threshold
        self.cache_images = cache_images
        self.prefetch_factor = prefetch_factor
        self.use_blurred_mask = use_blurred_mask
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
        watermarked_path = self.image_files[idx]
        image_name = os.path.basename(watermarked_path)
        
        # 使用缓存的图像或读取新图像
        if self.image_cache and idx in self.image_cache:
            watermarked_img = self.image_cache[idx]
        else:
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
    
    def _safe_imread(self, image_path, max_retries=2):
        """优化的安全图像读取"""
        for attempt in range(max_retries):
            try:
                # 首先检查文件是否存在且大小合理
                if not os.path.exists(image_path) or os.path.getsize(image_path) < 1024:
                    return None
                    
                # 使用OpenCV读取
                img = cv2.imread(image_path)
                if img is not None and img.size > 0:
                    # 验证图像形状是否合理
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        return img
                            
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(f"读取图片失败: {image_path}, 错误: {str(e)}")
                    
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
                        mask_dir = self.mask_dirs[0]
                        # 确保mask目录存在
                        os.makedirs(mask_dir, exist_ok=True)
                        mask_path = os.path.join(mask_dir, mask_name)
                        try:
                            cv2.imwrite(mask_path, mask)
                            logger.info(f"生成并保存mask: {mask_path}")
                        except Exception as e:
                            logger.warning(f"保存mask失败: {mask_path}, 错误: {str(e)}")
                    
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
        # 使用小核进行开运算去噪
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

      
        # 根据use_blurred_mask参数决定处理方式
        if self.use_blurred_mask:
            # 模糊mask：包含形态学操作去噪和连接断开的区域
            
            # 使用中等核进行闭运算连接断开的区域
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=3)
            
            # 使用更大的核进行强闭运算，确保连通性
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
            
            # 膨胀操作扩大mask区域
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask = cv2.dilate(mask, kernel_dilate, iterations=2)
            
            # 连通组件分析，保留最大的连通域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            if num_labels > 1:  # 有连通组件（除了背景）
                # 找到最大的连通组件（排除背景标签0）
                largest_component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                
                # 创建只包含最大连通组件的mask
                mask = (labels == largest_component_label).astype(np.uint8) * 255
                
                # 如果最大连通组件太小，则保留所有较大的组件
                max_area = stats[largest_component_label, cv2.CC_STAT_AREA]
                if max_area < 500:  # 如果最大组件面积小于阈值
                    mask = np.zeros_like(labels, dtype=np.uint8)
                    for i in range(1, num_labels):
                        if stats[i, cv2.CC_STAT_AREA] > 200:  # 保留面积大于200的组件
                            mask[labels == i] = 255
            
            # 轮廓检测和凸包处理以获得更完整的连通域
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 创建新的mask
                connected_mask = np.zeros_like(mask)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # 最小面积阈值
                        # 计算凸包以获得更完整的连通域
                        hull = cv2.convexHull(contour)
                        
                        # 如果凸包面积与原轮廓面积比值合理，使用凸包
                        hull_area = cv2.contourArea(hull)
                        if hull_area > 0 and area / hull_area > 0.6:  # 凸性比例阈值
                            cv2.fillPoly(connected_mask, [hull], 255)
                        else:
                            # 否则使用多边形近似
                            epsilon = 0.015 * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            cv2.fillPoly(connected_mask, [approx], 255)
                
                mask = connected_mask
            
            # 生成模糊的mask
            mask = self._apply_blur_to_mask(mask)
        else:
            # 精确mask：只进行简单的平滑处理，不包含形态学操作
            mask = cv2.GaussianBlur(mask, (3, 3), 0.5)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _apply_blur_to_mask(self, mask):
        """对mask应用模糊处理，生成软边缘的模糊mask"""
        # 首先进行轻微的高斯模糊
        blurred_mask = cv2.GaussianBlur(mask, (15, 15), 5.0)
        
        # 应用更强的模糊以创建渐变边缘
        blurred_mask = cv2.GaussianBlur(blurred_mask, (31, 31), 10.0)
        
        # 可选：添加一些随机噪声来增加训练的鲁棒性
        if self.mode == 'train':
            noise = np.random.normal(0, 5, blurred_mask.shape).astype(np.float32)
            blurred_mask = blurred_mask.astype(np.float32) + noise
            blurred_mask = np.clip(blurred_mask, 0, 255).astype(np.uint8)
        
        return blurred_mask

def get_transparent_watermark_transform(img_size=512):
    """专门针对透明水印的数据增强变换"""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), shear=(-5, 5), p=0.3),
        
        # 针对透明水印的特殊增强
        A.RandomBrightnessContrast(
            brightness_limit=0.3,  # 增大亮度变化
            contrast_limit=0.3,    # 增大对比度变化
            p=0.7  # 提高应用概率
        ),
        A.HueSaturationValue(
            hue_shift_limit=15, 
            sat_shift_limit=30,    # 增大饱和度变化
            val_shift_limit=20,    # 增大明度变化
            p=0.5
        ),
        
        # 添加噪声，模拟真实环境
        A.GaussNoise(p=0.3),
        
        # 模糊效果，模拟图像质量问题
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
        ], p=0.2),
        
        # JPEG压缩，模拟实际使用场景
        A.ImageCompression(quality_range=(60, 100), p=0.3),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], is_check_shapes=False)

def get_enhanced_train_transform(img_size=512):
    """增强版训练数据变换，包含透明水印优化"""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
        
        # 增强的亮度对比度调整
        A.RandomBrightnessContrast(
            brightness_limit=0.25, 
            contrast_limit=0.25, 
            p=0.6
        ),
        A.HueSaturationValue(
            hue_shift_limit=12, 
            sat_shift_limit=25, 
            val_shift_limit=15, 
            p=0.4
        ),
        
        # 添加更多增强技术
        A.OneOf([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ], p=0.3),
        
        # 噪声和模糊
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
        ], p=0.15),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], is_check_shapes=False)

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
    ], is_check_shapes=False)

def get_val_transform(img_size=512):
    """获取验证时的数据变换"""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], is_check_shapes=False)

def create_datasets(cfg, use_blurred_mask=False):
    """创建训练集和验证集
    
    Args:
        cfg: 配置对象
        use_blurred_mask (bool): 是否使用模糊的mask进行训练，默认False生成精确mask
    """
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

    # 根据配置选择数据增强策略
    augmentation_type = getattr(cfg.DATA, 'AUGMENTATION_TYPE', 'transparent_watermark')
    
    if augmentation_type == "transparent_watermark":
        train_transform = get_transparent_watermark_transform(cfg.DATA.IMG_SIZE)
        print(f"使用透明水印专用数据增强策略")
    elif augmentation_type == "enhanced":
        train_transform = get_enhanced_train_transform(cfg.DATA.IMG_SIZE)
        print(f"使用增强版数据增强策略")
    else:  # basic
        train_transform = get_train_transform(cfg.DATA.IMG_SIZE)
        print(f"使用基础数据增强策略")
    
    full_dataset = WatermarkDataset(
        watermarked_dirs=watermarked_dirs,
        clean_dirs=clean_dirs,
        mask_dirs=mask_dirs,
        transform=train_transform,
        mode='train',
        generate_mask_threshold=cfg.DATA.GENERATE_MASK_THRESHOLD,
        use_blurred_mask=use_blurred_mask
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
    
    # 创建训练集和验证集，使用不同的数据变换
    # 训练集使用透明水印专用增强变换
    train_dataset_full = WatermarkDataset(
        watermarked_dirs=watermarked_dirs,
        clean_dirs=clean_dirs,
        mask_dirs=mask_dirs,
        transform=train_transform,
        mode='train',
        generate_mask_threshold=cfg.DATA.GENERATE_MASK_THRESHOLD,
        use_blurred_mask=use_blurred_mask
    )
    
    # 验证集使用标准验证变换
    val_dataset_full = WatermarkDataset(
        watermarked_dirs=watermarked_dirs,
        clean_dirs=clean_dirs,
        mask_dirs=mask_dirs,
        transform=get_val_transform(cfg.DATA.IMG_SIZE),
        mode='val',
        generate_mask_threshold=cfg.DATA.GENERATE_MASK_THRESHOLD,
        use_blurred_mask=use_blurred_mask
    )
    
    # 创建子集
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    
    # 输出mask类型信息
    mask_type = "模糊mask" if use_blurred_mask else "精确mask"
    print(f"数据集划分完成:")
    print(f"训练集: {len(train_dataset)} 张图像")
    print(f"验证集: {len(val_dataset)} 张图像")
    print(f"Mask类型: {mask_type}")
    
    return train_dataset, val_dataset