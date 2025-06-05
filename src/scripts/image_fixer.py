import os
import cv2
import numpy as np
from PIL import Image, ImageFile
import logging
from tqdm import tqdm
import shutil
from pathlib import Path
import io
import warnings
from contextlib import redirect_stderr

# 不允许加载截断的图片，这样可以更好地检测损坏
ImageFile.LOAD_TRUNCATED_IMAGES = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageFixer:
    def __init__(self):
        self.corrupted_files = []
        self.fixed_files = []
        self.failed_files = []
        self.opencv_warnings = []
    
    def is_image_corrupted(self, image_path):
        """检测图片是否损坏 - 改进版本"""
        corrupted = False
        
        # 方法1: 检测OpenCV警告
        try:
            # 捕获stderr来检测OpenCV的警告信息
            stderr_capture = io.StringIO()
            with redirect_stderr(stderr_capture):
                img = cv2.imread(image_path)
            
            stderr_output = stderr_capture.getvalue()
            if "Corrupt JPEG data" in stderr_output or "bad Huffman code" in stderr_output:
                logger.warning(f"OpenCV检测到损坏: {image_path}")
                corrupted = True
            
            if img is None:
                logger.warning(f"OpenCV无法读取: {image_path}")
                return True
                
        except Exception as e:
            logger.warning(f"OpenCV读取异常: {image_path}, 错误: {str(e)}")
            corrupted = True
        
        # 方法2: 严格的PIL检测
        try:
            # 临时禁用截断图片加载
            original_setting = ImageFile.LOAD_TRUNCATED_IMAGES
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                with Image.open(image_path) as pil_img:
                    pil_img.verify()  # 验证图片完整性
                
                # 检查是否有警告
                if w:
                    for warning in w:
                        if "truncated" in str(warning.message).lower() or "corrupt" in str(warning.message).lower():
                            logger.warning(f"PIL检测到问题: {image_path}, 警告: {warning.message}")
                            corrupted = True
            
            # 方法3: 尝试完全加载图片数据
            with Image.open(image_path) as pil_img:
                pil_img.load()  # 强制加载所有图片数据
                # 尝试转换为numpy数组，这会强制解码所有数据
                np.array(pil_img)
            
            # 恢复原设置
            ImageFile.LOAD_TRUNCATED_IMAGES = original_setting
            
        except Exception as e:
            logger.warning(f"PIL检测到损坏: {image_path}, 错误: {str(e)}")
            corrupted = True
        
        # 方法4: 文件完整性检查
        try:
            with open(image_path, 'rb') as f:
                # 检查JPEG文件是否以正确的标记结束
                f.seek(-2, 2)  # 移动到文件末尾前2字节
                last_bytes = f.read(2)
                if image_path.lower().endswith(('.jpg', '.jpeg')):
                    if last_bytes != b'\xff\xd9':  # JPEG结束标记
                        logger.warning(f"JPEG文件结束标记缺失: {image_path}")
                        corrupted = True
        except Exception as e:
            logger.warning(f"文件完整性检查失败: {image_path}, 错误: {str(e)}")
            corrupted = True
        
        return corrupted
    
    def fix_image(self, image_path, backup_dir=None):
        """修复损坏的图片"""
        try:
            # 创建备份
            if backup_dir:
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, os.path.basename(image_path))
                shutil.copy2(image_path, backup_path)
                logger.info(f"已备份原文件到: {backup_path}")
            
            # 尝试使用PIL修复
            # 临时允许加载截断图片进行修复
            original_setting = ImageFile.LOAD_TRUNCATED_IMAGES
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            
            try:
                with Image.open(image_path) as img:
                    # 转换为RGB模式（如果需要）
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 重新保存图片，这会重新编码并可能修复一些问题
                    temp_path = image_path + '.tmp'
                    img.save(temp_path, 'JPEG', quality=95, optimize=True)
                    
                    # 验证修复后的图片
                    if not self.is_image_corrupted(temp_path):
                        # 替换原文件
                        shutil.move(temp_path, image_path)
                        logger.info(f"成功修复: {image_path}")
                        return True
                    else:
                        os.remove(temp_path)
                        return False
            finally:
                # 恢复原设置
                ImageFile.LOAD_TRUNCATED_IMAGES = original_setting
                    
        except Exception as e:
            logger.error(f"修复失败: {image_path}, 错误: {str(e)}")
            return False
    
    def scan_directory(self, directory, extensions=None):
        """扫描目录中的所有图片文件"""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        image_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_files.append(os.path.join(root, file))
        
        return image_files
    
    def fix_directory(self, directory, backup_dir=None, extensions=None):
        """修复目录中所有损坏的图片"""
        logger.info(f"开始扫描目录: {directory}")
        
        # 获取所有图片文件
        image_files = self.scan_directory(directory, extensions)
        logger.info(f"找到 {len(image_files)} 个图片文件")
        
        # 检测损坏的图片
        logger.info("检测损坏的图片...")
        for image_path in tqdm(image_files, desc="检测图片"):
            if self.is_image_corrupted(image_path):
                self.corrupted_files.append(image_path)
        
        logger.info(f"发现 {len(self.corrupted_files)} 个损坏的图片")
        
        if not self.corrupted_files:
            logger.info("没有发现损坏的图片")
            return
        
        # 显示前10个损坏的文件
        logger.info("损坏的图片示例:")
        for i, corrupted_file in enumerate(self.corrupted_files[:10]):
            logger.info(f"  {i+1}. {corrupted_file}")
        if len(self.corrupted_files) > 10:
            logger.info(f"  ... 还有 {len(self.corrupted_files) - 10} 个")
        
        # 修复损坏的图片
        logger.info("开始修复损坏的图片...")
        for image_path in tqdm(self.corrupted_files, desc="修复图片"):
            if self.fix_image(image_path, backup_dir):
                self.fixed_files.append(image_path)
            else:
                self.failed_files.append(image_path)
        
        # 打印统计信息
        self.print_summary()
    
    def print_summary(self):
        """打印修复统计信息"""
        logger.info("\n=== 图片修复统计 ===")
        logger.info(f"发现损坏图片: {len(self.corrupted_files)}")
        logger.info(f"成功修复: {len(self.fixed_files)}")
        logger.info(f"修复失败: {len(self.failed_files)}")
        
        if self.failed_files:
            logger.warning("\n修复失败的文件:")
            for file_path in self.failed_files:
                logger.warning(f"  - {file_path}")
            logger.warning("建议手动检查这些文件或从备份中恢复")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='检测和修复损坏的图片文件')
    parser.add_argument('directory', help='要扫描的目录路径')
    parser.add_argument('--backup-dir', help='备份目录路径（可选）')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
                       help='要处理的文件扩展名')
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.directory):
        logger.error(f"目录不存在: {args.directory}")
        return
    
    # 创建图片修复器
    fixer = ImageFixer()
    
    # 开始修复
    fixer.fix_directory(
        directory=args.directory,
        backup_dir=args.backup_dir,
        extensions=args.extensions
    )

if __name__ == '__main__':
    main()