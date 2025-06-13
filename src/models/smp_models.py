"""
基于SMP库的模型定义模块
提供统一的模型创建接口，支持多种分割模型
"""
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from typing import Optional, Union, List

class SMPModelFactory:
    """
    SMP模型工厂类
    支持创建各种预训练的分割模型
    """
    
    # 支持的模型列表
    SUPPORTED_MODELS = {
        'Unet': smp.Unet,
        'UnetPlusPlus': smp.UnetPlusPlus,
        'MAnet': smp.MAnet,
        'Linknet': smp.Linknet,
        'FPN': smp.FPN,
        'PSPNet': smp.PSPNet,
        'PAN': smp.PAN,
        'DeepLabV3': smp.DeepLabV3,
        'DeepLabV3Plus': smp.DeepLabV3Plus,
    }
    
    @classmethod
    def create_model(
        cls,
        model_name: str,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        **kwargs
    ) -> nn.Module:
        """
        创建SMP模型
        
        Args:
            model_name: 模型名称
            encoder_name: 编码器名称
            encoder_weights: 预训练权重
            in_channels: 输入通道数
            classes: 输出类别数
            activation: 激活函数
            **kwargs: 其他模型参数
            
        Returns:
            创建的模型实例
        """
        if model_name not in cls.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {list(cls.SUPPORTED_MODELS.keys())}"
            )
        
        model_class = cls.SUPPORTED_MODELS[model_name]
        
        # 创建模型
        model = model_class(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            **kwargs
        )
        
        return model
    
    @classmethod
    def get_available_encoders(cls) -> List[str]:
        """获取可用的编码器列表"""
        return smp.encoders.get_encoder_names()
    
    @classmethod
    def get_encoder_info(cls, encoder_name: str) -> dict:
        """获取编码器信息"""
        try:
            encoder = smp.encoders.get_encoder(encoder_name)
            return {
                'name': encoder_name,
                'params': encoder.params,
                'out_channels': encoder.out_channels,
            }
        except Exception as e:
            return {'error': str(e)}

def create_model_from_config(cfg) -> nn.Module:
    """
    从配置创建模型
    
    Args:
        cfg: 配置对象
        
    Returns:
        创建的模型实例
    """
    # 基础参数
    model_params = {
        'model_name': cfg.MODEL.NAME,
        'encoder_name': cfg.MODEL.ENCODER_NAME,
        'encoder_weights': cfg.MODEL.ENCODER_WEIGHTS,
        'in_channels': cfg.MODEL.IN_CHANNELS,
        'classes': cfg.MODEL.CLASSES,
        'activation': cfg.MODEL.ACTIVATION,
    }
    
    # 添加可选的架构参数
    if hasattr(cfg.MODEL, 'ENCODER_DEPTH'):
        model_params['encoder_depth'] = cfg.MODEL.ENCODER_DEPTH
    
    if hasattr(cfg.MODEL, 'DECODER_CHANNELS'):
        model_params['decoder_channels'] = cfg.MODEL.DECODER_CHANNELS
    
    return SMPModelFactory.create_model(**model_params)

# 为了兼容性，提供一个简单的包装类
class WatermarkSegmentationModel(nn.Module):
    """
    水印分割模型包装类
    提供统一的接口和额外的功能
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = create_model_from_config(cfg)
        
    def forward(self, x):
        """前向传播"""
        return self.model(x)
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.cfg.MODEL.NAME,
            'encoder_name': self.cfg.MODEL.ENCODER_NAME,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_channels': self.cfg.MODEL.IN_CHANNELS,
            'output_classes': self.cfg.MODEL.CLASSES,
        }