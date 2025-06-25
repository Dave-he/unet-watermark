import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random
import os
from tqdm import tqdm

class EasyOCRDetector:
    """
    文字区域mask生成器类
    使用EasyOCR检测图片中的文字区域并生成二值化mask图
    """
    
    def __init__(self, languages=None, gpu=True, verbose=True):
        """
        初始化文字mask生成器
        
        Args:
            languages: 支持的语言列表，None表示使用所有支持的语言
            gpu: 是否使用GPU加速
            verbose: 是否显示详细信息
        """
        self.gpu = gpu
        self.verbose = verbose
        self.reader = None
        
        # 设置支持的语言
        if languages is None:
            # EasyOCR常用语言列表
            self.languages = ['en', 'ch_sim']
            #self.languages = ['ja', 'ko', 'fr', 'de', 'es', 'pt', 'ru']
            if self.verbose:
                print(f"使用默认的{len(self.languages)}种语言: {', '.join(self.languages[:5])}...")
        else:
            self.languages = languages
            if self.verbose:
                print(f"使用指定的{len(self.languages)}种语言: {', '.join(self.languages)}")
        
        # 支持的图片格式
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def _init_reader(self):
        """延迟初始化EasyOCR读取器"""
        if self.reader is None:
            if self.verbose:
                print("正在初始化EasyOCR读取器...")
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
    
    def generate_text_mask(self, image_path, output_path=None, visualize=False):
        """
        为单张图片生成文字区域mask
        
        Args:
            image_path: 输入图片路径
            output_path: 输出mask路径，None表示不保存
            visualize: 是否显示可视化结果
            
        Returns:
            binary_mask: 二值化mask图像
        """
        # 读取图片
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 初始化读取器
        self._init_reader()
        
        # 检测文字区域
        if self.verbose:
            print("正在检测多语言文字区域...")
        results = self.reader.readtext(img)
        
        # 创建mask图
        mask = np.zeros_like(img_rgb)
        for bbox, text, conf in results:
            pts = np.array(bbox, dtype=np.int32)
            cv2.fillPoly(mask, [pts], (255, 255, 255))
        
        # 生成二值化mask
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
        
        # 保存与可视化
        if output_path:
            cv2.imwrite(str(output_path), binary_mask)
        
        if visualize:
            self._visualize_results(img_rgb, mask, binary_mask)
        
        return binary_mask
    
    def detect_text_regions(self, image_path, languages=None):
        """
        检测图片中的文字区域
        
        Args:
            image_path: 输入图片路径
            languages: 支持的语言列表，None表示使用默认语言
            
        Returns:
            list: 文字区域列表，每个元素包含bbox信息
        """
        # 读取图片
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 初始化读取器
        self._init_reader()
        
        # 检测文字区域
        results = self.reader.readtext(img)
        
        # 转换为统一格式
        text_regions = []
        for bbox, text, conf in results:
            # 将bbox转换为统一格式 [x1, y1, x2, y2, x3, y3, x4, y4]
            if len(bbox) == 4 and len(bbox[0]) == 2:
                # bbox是四个点的坐标
                flat_bbox = [coord for point in bbox for coord in point]
                text_regions.append({
                    'bbox': flat_bbox,
                    'text': text,
                    'confidence': conf
                })
        
        return text_regions
    
    def _visualize_results(self, img_rgb, mask, binary_mask):
        """可视化处理结果"""
        plt.figure(figsize=(15, 5))
        plt.subplot(131), plt.title('原图'), plt.imshow(img_rgb), plt.axis('off')
        plt.subplot(132), plt.title('文字区域'), plt.imshow(mask), plt.axis('off')
        plt.subplot(133), plt.title('二值化Mask'), plt.imshow(binary_mask, cmap='gray'), plt.axis('off')
        plt.tight_layout(), plt.show()
    
    def _get_image_files(self, input_folder):
        """获取文件夹中的所有图片文件"""
        input_path = Path(input_folder)
        image_files = []
        
        for ext in self.image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        return image_files
    
    def batch_process(self, input_folder, output_folder, limit=None, visualize=False, random_seed=42):
        """
        批量处理文件夹中的图片，生成文字区域mask
        
        Args:
            input_folder: 输入图片文件夹路径
            output_folder: 输出mask文件夹路径
            limit: 随机选取的图片数量限制，None表示处理所有图片
            visualize: 是否显示可视化结果
            random_seed: 随机种子
            
        Returns:
            dict: 包含处理结果统计的字典
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        if not input_path.exists():
            raise ValueError(f"输入文件夹不存在: {input_folder}")
        
        # 创建输出文件夹
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图片文件
        image_files = self._get_image_files(input_folder)
        
        if not image_files:
            if self.verbose:
                print(f"在文件夹 {input_folder} 中未找到支持的图片文件")
            return {'success': 0, 'error': 0, 'total': 0}
        
        if self.verbose:
            print(f"找到 {len(image_files)} 张图片")
        
        # 随机选取指定数量的图片
        if limit and limit < len(image_files):
            random.seed(random_seed)
            image_files = random.sample(image_files, limit)
            if self.verbose:
                print(f"随机选取 {limit} 张图片进行处理")
        
        # 批量处理
        success_count = 0
        error_count = 0
        
        for image_file in tqdm(image_files, desc="处理图片", disable=not self.verbose):
            try:
                # 生成输出文件名
                output_file = output_path / f"{image_file.stem}{image_file.suffix}"
                
                # 处理图片
                self.generate_text_mask(
                    image_file, 
                    output_file, 
                    visualize=visualize
                )
                
                success_count += 1
                if self.verbose:
                    print(f"✓ 处理完成: {image_file.name} -> {output_file.name}")
                
            except Exception as e:
                error_count += 1
                if self.verbose:
                    print(f"✗ 处理失败: {image_file.name}, 错误: {str(e)}")
        
        result = {
            'success': success_count,
            'error': error_count,
            'total': len(image_files)
        }
        
        if self.verbose:
            print(f"\n处理完成! 成功: {success_count}, 失败: {error_count}")
        
        return result

# 向后兼容的函数接口
def generate_text_mask_all_languages(image_path, output_path=None, visualize=False):
    """
    向后兼容的函数接口
    使用EasyOCR检测图片中所有支持的语言文字区域并生成二值化mask图
    """
    generator = TextMaskGenerator()
    return generator.generate_text_mask(image_path, output_path, visualize)

def batch_process_images(input_folder, output_folder, limit=None, visualize=False):
    """
    向后兼容的函数接口
    批量处理文件夹中的图片，生成文字区域mask
    """
    generator = TextMaskGenerator()
    return generator.batch_process(input_folder, output_folder, limit, visualize)

def main():
    parser = argparse.ArgumentParser(description='批量生成图片文字区域mask')
    parser.add_argument('input_folder', help='输入图片文件夹路径')
    parser.add_argument('output_folder', help='输出mask文件夹路径')
    parser.add_argument('--limit', type=int, help='随机选取的图片数量限制')
    parser.add_argument('--visualize', action='store_true', help='显示可视化结果')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，用于可重现的随机选择')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    try:
        batch_process_images(
            args.input_folder,
            args.output_folder,
            limit=args.limit,
            visualize=args.visualize
        )
    except Exception as e:
        print(f"错误: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())