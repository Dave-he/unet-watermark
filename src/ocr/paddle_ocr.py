# https://www.paddlepaddle.org.cn/en/install/quick


import base64
import os
from threading import Thread
import requests
import os
import glob
import random
from PIL import Image, ImageDraw
import numpy as np
from typing import Optional, List, Dict, Any


class PaddleOCRProcessor:
    """PaddleOCR处理器类，用于OCR识别和mask生成"""
    
    def __init__(self, api_url: str = "http://localhost:8080/ocr", 
                 input_dir: str = "./input", 
                 output_dir: str = "./output"):
        """
        初始化PaddleOCR处理器
        
        Args:
            api_url: OCR API的URL地址
            input_dir: 输入图片目录
            output_dir: 输出结果目录
        """
        self.api_url = api_url
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # 支持的图片格式
        self.image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_mask_image(self, image_path: str, ocr_result: Dict[str, Any], output_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        根据OCR结果创建文字区域的mask二值化图
        
        Args:
            image_path: 输入图片路径
            output_path: 输出mask图片路径，如果为None则不保存
            
        Returns:
            mask图像的numpy数组，如果失败返回None
        """
        try:
            # 读取图片
            with Image.open(image_path) as img:
                width, height = img.size
                image_array = np.array(img)
            
            if not ocr_result:
                # 发送OCR请求
                ocr_result = self.ocr_request(image_path)
                if not ocr_result:
                    print(f"OCR识别失败或无文字: {image_path}")
                    # 返回空的mask而不是None，保持与EasyOCR一致的行为
                    return np.zeros((height, width), dtype=np.uint8)
            
            # 创建mask图像
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # 从ocrResults中获取坐标数据
            polys_to_use = None
            if 'ocrResults' in ocr_result and ocr_result['ocrResults']:
                pruned_result = ocr_result['ocrResults'][0].get('prunedResult', {})
                
                # 优先使用dt_polys（检测到的文字区域多边形）
                if 'dt_polys' in pruned_result and pruned_result['dt_polys']:
                    polys_to_use = pruned_result['dt_polys']

                elif 'rec_polys' in pruned_result and pruned_result['rec_polys']:
                    polys_to_use = pruned_result['rec_polys']
                    
                elif 'rec_boxes' in pruned_result and pruned_result['rec_boxes']:
                    # 如果只有矩形框，转换为多边形
                    polys_to_use = []
                    for box in pruned_result['rec_boxes']:
                        # box格式: [x1, y1, x2, y2]
                        x1, y1, x2, y2 = box
                        poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        polys_to_use.append(poly)
            
            if polys_to_use:
                for i, poly in enumerate(polys_to_use):
                    # 将多边形坐标转换为numpy数组
                    points = np.array([[int(point[0]), int(point[1])] for point in poly], dtype=np.int32)
                    # 使用PIL绘制多边形到mask
                    mask_img = Image.fromarray(mask)
                    draw = ImageDraw.Draw(mask_img)
                    flat_points = [coord for point in points for coord in point]
                    draw.polygon(flat_points, fill=255)
                    mask = np.array(mask_img)
                    print(f"Drew polygon {i+1}: {poly}")
            else:
                print("No polygon data found in OCR result")
            
            # 保存mask图像
            if output_path:
                mask_img = Image.fromarray(mask)
                mask_img.save(output_path)
                print(f"Mask图像已保存: {output_path}")
            
            return mask
            
        except Exception as e:
            print(f"创建mask图像失败: {str(e)}")
            # 返回空的mask而不是None，保持一致性
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    return np.zeros((height, width), dtype=np.uint8)
            except:
                pass
            return None
    
    def ocr_request(self, image_path: str) -> Optional[Dict[str, Any]]:
        """向OCR API发送请求
        
        Args:
            image_path: 图片路径
            
        Returns:
            OCR识别结果，失败时返回None
        """
        # 读取图片文件
        with open(image_path, "rb") as file:
            file_bytes = file.read()
            file_data = base64.b64encode(file_bytes).decode("ascii")
        
        payload = {"file": file_data, "fileType": 1}
        
        try:
            response = requests.post(self.api_url, json=payload)
            
            if response.status_code == 200:
                return response.json()["result"]
            else:
                print(f"Error processing {image_path}: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def process_single_image(self, image_path: str, save_mask: bool = True, save_result: bool = True) -> Optional[Image.Image]:
        """处理单张图片
        
        Args:
            image_path: 图片路径
            save_mask: 是否保存mask图像到文件
            
        Returns:
            生成的mask图像，失败时返回None
        """
        print(f"Processing: {image_path}")
        
        # 发送OCR请求
        ocr_result = self.ocr_request(image_path)
        if ocr_result is None:
            return None
        
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        if save_result:
            os.makedirs(os.path.join(self.output_dir, "res"), exist_ok=True)
            for i, res in enumerate(ocr_result["ocrResults"]):
                print(res["prunedResult"])
                ocr_img_path=  os.path.join(self.output_dir, "res", f"{base_name}_{i}.jpg")
                with open(ocr_img_path, "wb") as f:
                    f.write(base64.b64decode(res["ocrImage"]))
                print(f"Output image saved at {ocr_img_path}")

        # 创建mask图像
        mask_array = self.create_mask_image(image_path, ocr_result)
        
        if save_mask and mask_array is not None:
            os.makedirs(os.path.join(self.output_dir, "mask"), exist_ok=True)
            mask_path = os.path.join(self.output_dir, "mask", f"{base_name}.png")
            # 将numpy数组转换为PIL Image并保存
            mask_image = Image.fromarray(mask_array)
            mask_image.save(mask_path)
            print(f"Mask image saved: {mask_path}")
            return mask_image
        elif mask_array is not None:
            # 如果不保存文件，仍然返回PIL Image对象
            return Image.fromarray(mask_array)
        
        return None
    
    def get_image_files(self, directory: Optional[str] = None) -> List[str]:
        """获取目录下所有支持的图片文件
        
        Args:
            directory: 目录路径，默认使用input_dir
            
        Returns:
            图片文件路径列表
        """
        if directory is None:
            directory = self.input_dir
            
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(glob.glob(os.path.join(directory, ext)))
            image_files.extend(glob.glob(os.path.join(directory, ext.upper())))
        
        return image_files
    
    def batch_process(self, input_directory: Optional[str] = None, limit: Optional[int] = None) -> None:
        """
        批量处理目录下的所有图片
        
        Args:
            input_directory: 输入目录，默认使用self.input_dir
            limit: 限制处理的图片数量，如果为None则处理所有图片
        """
        if input_directory is None:
            input_directory = self.input_dir
            
        print(f"Starting batch OCR processing...")
        print(f"Input directory: {input_directory}")
        print(f"Output directory: {self.output_dir}")
        
        # 获取所有支持的图片文件
        image_files = self.get_image_files(input_directory)
        
        if not image_files:
            print("No image files found in input directory.")
            return
        
        # 如果设置了limit参数，随机选择指定数量的图片
        if limit is not None and limit > 0:
            if limit < len(image_files):
                image_files = random.sample(image_files, limit)
                print(f"Randomly selected {limit} image(s) from {len(self.get_image_files(input_directory))} total images")
            else:
                print(f"Limit ({limit}) is greater than available images ({len(image_files)}), processing all images")
        
        print(f"Found {len(image_files)} image(s) to process:")
        for img_file in image_files:
            print(f"  - {os.path.basename(img_file)}")
        print()
        
        import time
        # 处理每张图片
        for image_file in image_files:
            time.sleep(0.1)
            self.process_single_image(image_file)
        
        print("Batch processing completed!")



def main():
    """
    主函数：批量处理图片
    
    Args:
        input_dir (str, optional): 输入目录路径，默认为 './input'
        output_dir (str, optional): 输出目录路径，默认为 './output'
        api_url (str, optional): OCR API地址，默认为 'http://localhost:8080/ocr'
        limit (int, optional): 随机选择处理的图片数量，默认处理所有图片
    """

    import argparse
    
    parser = argparse.ArgumentParser(description='PaddleOCR批量处理工具')
    parser.add_argument('--input', '-i', type=str, default='/Users/hyx/unet-watermark/data/test', 
                       help='输入目录路径 (默认: ./input)')
    parser.add_argument('--output', '-o', type=str, default='./output',
                       help='输出目录路径 (默认: ./output)')
    parser.add_argument('--api-url', '-u', type=str, default='http://localhost:8080/ocr',
                       help='OCR API地址')
    parser.add_argument('--limit', '-l', type=int, default=10,
                       help='随机选择处理的图片数量')
    
    args = parser.parse_args()

    processor = PaddleOCRProcessor(input_dir=args.input, output_dir=args.output, api_url=args.api_url)
    processor.batch_process(limit=args.limit)


if __name__ == "__main__":
    main()