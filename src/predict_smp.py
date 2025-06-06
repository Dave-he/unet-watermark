import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import subprocess
import tempfile
import argparse
import logging
from pathlib import Path

# 导入自定义模块
from configs.config import get_cfg_defaults, update_config
from models.smp_models import create_model_from_config
from utils.dataset import get_val_transform

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WatermarkPredictor:
    """水印预测器类"""
    
    def __init__(self, model_path, config_path=None, config=None, device='cpu'):
        """初始化预测器"""
        self.device = torch.device(device)
        
        # 加载配置
        if config is not None:
            self.cfg = config
        else:
            self.cfg = get_cfg_defaults()
            if config_path and os.path.exists(config_path):
                update_config(self.cfg, config_path)
        
        # 加载模型
        self.model, self.model_info = self._load_model(model_path)
        
        # 数据变换
        self.transform = get_val_transform(self.cfg.DATA.IMG_SIZE)
        
        logger.info(f"预测器初始化完成，使用设备: {self.device}")
        self._print_model_info()
    
    def _load_model(self, model_path):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 创建模型
        model = create_model_from_config(self.cfg).to(self.device)
        
        # 加载权重 - 添加 weights_only=False 以兼容包含配置对象的模型文件
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"从检查点加载模型: {model_path}")
            model_info = {
                'epoch': checkpoint.get('epoch', 'Unknown'),
                'val_loss': checkpoint.get('val_loss', 'Unknown'),
                'val_metrics': checkpoint.get('val_metrics', {}),
                'train_loss': checkpoint.get('train_loss', 'Unknown'),
                'train_metrics': checkpoint.get('train_metrics', {}),
                'best_val_loss': checkpoint.get('best_val_loss', 'Unknown'),
                'is_final': checkpoint.get('is_final', False)
            }
        else:
            model.load_state_dict(checkpoint)
            logger.info(f"加载模型权重: {model_path}")
            model_info = {'epoch': 'Unknown', 'val_loss': 'Unknown'}
        
        model.eval()
        return model, model_info
    
    def _print_model_info(self):
        """打印模型信息"""
        logger.info("=" * 50)
        logger.info("模型信息:")
        logger.info(f"训练轮数: {self.model_info.get('epoch', 'Unknown')}")
        logger.info(f"验证损失: {self.model_info.get('val_loss', 'Unknown')}")
        logger.info(f"训练损失: {self.model_info.get('train_loss', 'Unknown')}")
        
        val_metrics = self.model_info.get('val_metrics', {})
        if val_metrics:
            logger.info(f"验证IoU: {val_metrics.get('iou', 'Unknown')}")
            logger.info(f"验证F1: {val_metrics.get('f1', 'Unknown')}")
        
        if self.model_info.get('is_final', False):
            logger.info("模型类型: 最终模型")
        elif self.model_info.get('best_val_loss') == self.model_info.get('val_loss'):
            logger.info("模型类型: 最佳模型")
        else:
            logger.info("模型类型: 检查点模型")
        logger.info("=" * 50)
    
    def predict_mask(self, image_path):
        """
        预测单张图像的水印掩码
        
        Args:
            image_path (str): 图像路径
            
        Returns:
            np.ndarray: 预测的掩码
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用变换
        augmented = self.transform(image=image_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # 模型推理
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output)
        
        # 处理掩码
        mask = prob.cpu().numpy()[0, 0]  # 移除批次和通道维度
        mask = cv2.resize(mask, original_size)
        
        # 二值化掩码
        binary_mask = (mask > self.cfg.PREDICT.THRESHOLD).astype(np.uint8) * 255
        
        # 后处理
        if self.cfg.PREDICT.POST_PROCESS:
            binary_mask = self._post_process_mask(binary_mask)
        
        return binary_mask
    
    def _post_process_mask(self, mask):
        """掩码后处理"""
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 去除小连通域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 100  # 最小连通域面积
        
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                cv2.fillPoly(mask, [contour], 0)
        
        return mask
    
    def remove_watermark_with_iopaint(self, image_path, mask, output_path, model_name='lama'):
        """
        使用iopaint去除水印
        
        Args:
            image_path (str): 输入图像路径
            mask (np.ndarray): 掩码
            output_path (str): 输出路径
            model_name (str): iopaint模型名称
            
        Returns:
            tuple: (成功标志, 消息)
        """
        try:
            # 创建临时掩码文件
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_mask:
                cv2.imwrite(temp_mask.name, mask)
                temp_mask_path = temp_mask.name
            
            # 准备iopaint命令
            cmd = [
                'iopaint', 'run',
                '--model', model_name,
                '--device', self.device.type,
                '--input', image_path,
                '--mask', temp_mask_path,
                '--output', output_path
            ]
            
            # 运行iopaint
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # 清理临时文件
            os.unlink(temp_mask_path)
            
            return True, "成功"
            
        except subprocess.CalledProcessError as e:
            return False, f"IOPaint错误: {e.stderr}"
        except Exception as e:
            return False, f"错误: {str(e)}"
    
    def process_single_image(self, image_path, output_dir, save_mask=True, remove_watermark=False, iopaint_model='lama'):
        """
        处理单张图像
        
        Args:
            image_path (str): 输入图像路径
            output_dir (str): 输出目录
            save_mask (bool): 是否保存掩码
            remove_watermark (bool): 是否去除水印
            iopaint_model (str): iopaint模型名称
            
        Returns:
            dict: 处理结果
        """
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取基础文件名
            base_name = Path(image_path).stem
            
            # 预测掩码
            logger.info(f"预测掩码: {image_path}")
            mask = self.predict_mask(image_path)
            
            result = {
                'status': 'success',
                'input': image_path,
                'mask': None,
                'output': None
            }
            
            # 保存掩码
            if save_mask:
                mask_path = os.path.join(output_dir, f"{base_name}.png")
                cv2.imwrite(mask_path, mask)
                result['mask'] = mask_path
                logger.info(f"掩码已保存: {mask_path}")
            
            # 去除水印
            if remove_watermark:
                output_path = os.path.join(output_dir, f"{base_name}_restored.png")
                logger.info(f"使用iopaint去除水印...")
                
                success, message = self.remove_watermark_with_iopaint(
                    image_path, mask, output_path, iopaint_model
                )
                
                if success:
                    result['output'] = output_path
                    logger.info(f"水印去除成功: {output_path}")
                else:
                    result['status'] = 'partial'
                    result['error'] = message
                    logger.warning(f"水印去除失败: {message}")
            
            return result
            
        except Exception as e:
            logger.error(f"处理图像失败 {image_path}: {str(e)}")
            return {
                'status': 'error',
                'input': image_path,
                'error': str(e)
            }
    
    def process_batch(self, input_path, output_dir, save_mask=True, remove_watermark=False, iopaint_model='lama'):
        """
        批量处理图像
        
        Args:
            input_path (str): 输入路径（文件或目录）
            output_dir (str): 输出目录
            save_mask (bool): 是否保存掩码
            remove_watermark (bool): 是否去除水印
            iopaint_model (str): iopaint模型名称
            
        Returns:
            list: 处理结果列表
        """
        # 获取图像路径列表
        image_paths = []
        if os.path.isdir(input_path):
            for filename in os.listdir(input_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_paths.append(os.path.join(input_path, filename))
        elif os.path.isfile(input_path):
            if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(input_path)
        
        if not image_paths:
            logger.warning(f"在 {input_path} 中未找到图像文件")
            return []
        
        logger.info(f"开始处理 {len(image_paths)} 张图像...")
        
        results = []
        for image_path in tqdm(image_paths, desc="处理图像"):
            result = self.process_single_image(
                image_path, output_dir, save_mask, remove_watermark, iopaint_model
            )
            results.append(result)
        
        # 打印统计信息
        successful = sum(1 for r in results if r['status'] == 'success')
        partial = sum(1 for r in results if r['status'] == 'partial')
        failed = sum(1 for r in results if r['status'] == 'error')
        
        logger.info(f"\n处理完成！")
        logger.info(f"成功: {successful}")
        logger.info(f"部分成功: {partial}")
        logger.info(f"失败: {failed}")
        
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SMP UNet++ 水印检测预测')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径')
    parser.add_argument('--config', type=str, default='src/configs/unet_watermark.yaml', help='配置文件路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像路径或目录')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--device', type=str, default='cpu', help='设备类型')
    parser.add_argument('--batch-size', type=int, default=8, help='批处理大小')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值')
    parser.add_argument('--save-mask', action='store_true', help='保存掩码')
    parser.add_argument('--remove-watermark', action='store_true', help='去除水印')
    parser.add_argument('--iopaint-model', type=str, default='lama', help='IOPaint模型')
    parser.add_argument('--list-checkpoints', type=str, help='列出指定目录的检查点')
    
    args = parser.parse_args()
    
    # 列出检查点
    if args.list_checkpoints:
        from utils.model_manager import list_checkpoints
        list_checkpoints(args.list_checkpoints)
        return
    
    # 验证模型文件
    if not os.path.exists(args.model):
        logger.error(f"模型文件不存在: {args.model}")
        return
    
    # 创建预测器
    predictor = WatermarkPredictor(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )
    
    # 检查iopaint是否安装（如果需要去除水印）
    if args.remove_watermark:
        try:
            subprocess.run(['iopaint', '--help'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("错误: iopaint未安装。请使用以下命令安装:")
            logger.error("pip install iopaint")
            return
    
    try:
        # 初始化预测器
        predictor = WatermarkPredictor(args.model, args.config, args.device)
        
        # 覆盖阈值配置
        if args.threshold is not None:
            predictor.cfg.PREDICT.THRESHOLD = args.threshold
        
        # 处理图像
        results = predictor.process_batch(
            args.input, 
            args.output, 
            args.save_mask, 
            args.remove_watermark, 
            args.iopaint_model
        )
        
        # 保存结果摘要
        import json
        summary_path = os.path.join(args.output, 'processing_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"处理摘要已保存: {summary_path}")
        
    except Exception as e:
        logger.error(f"预测器初始化失败: {str(e)}")

if __name__ == "__main__":
    main()