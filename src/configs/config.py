"""
配置管理模块 - 使用YACS进行配置管理
支持从YAML文件加载配置，便于实验管理
"""
from yacs.config import CfgNode as CN

# 创建配置节点
_C = CN()

# 设备配置
_C.DEVICE = "cpu"  # "cuda" or "cpu"

# 模型配置
_C.MODEL = CN()
_C.MODEL.NAME = "UnetPlusPlus"  # smp支持的模型名称
_C.MODEL.ENCODER_NAME = "resnet34"  # 编码器骨干网络
_C.MODEL.ENCODER_WEIGHTS = "imagenet"  # 预训练权重
_C.MODEL.IN_CHANNELS = 3  # 输入通道数
_C.MODEL.CLASSES = 1  # 输出类别数
_C.MODEL.ACTIVATION = None  # 激活函数，None表示不使用

# 数据配置
_C.DATA = CN()
_C.DATA.ROOT_DIR = "./data/train"  # 数据集根目录
_C.DATA.ADDITIONAL_ROOT_DIRS = [] # 额外的数据集根目录列表
_C.DATA.IMG_SIZE = 512  # 图像大小
_C.DATA.GENERATE_MASK_THRESHOLD = 30  # 生成掩码的阈值
_C.DATA.TRAIN_RATIO = 0.8  # 训练集比例
_C.DATA.VAL_RATIO = 0.2  # 验证集比例
_C.DATA.SHUFFLE = True  # 是否打乱数据集
_C.DATA.SEED = 42  # 随机种子
_C.DATA.NUM_WORKERS = 4  # 数据加载器工作进程数

# 训练配置
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 16  # 批次大小
_C.TRAIN.EPOCHS = 300  # 训练轮数
_C.TRAIN.LR = 0.0001  # 学习率
_C.TRAIN.WEIGHT_DECAY = 0.0001  # 权重衰减
_C.TRAIN.OUTPUT_DIR = "./../logs/output"  # 输出目录
_C.TRAIN.MODEL_SAVE_PATH = "./../models/unet_watermark.pth"  # 模型保存路径
_C.TRAIN.LOG_INTERVAL = 10  # 日志打印间隔
_C.TRAIN.SAVE_INTERVAL = 5  # 模型保存间隔
_C.TRAIN.EARLY_STOPPING_PATIENCE = 10  # 早停耐心值

# 损失函数配置
_C.LOSS = CN()
_C.LOSS.NAME = "DiceLoss"  # 损失函数名称
_C.LOSS.MODE = "binary"  # 损失函数模式
_C.LOSS.SMOOTH = 1e-5  # 损失函数平滑项
_C.LOSS.BCE_WEIGHT = 0.5  # BCE损失权重
_C.LOSS.DICE_SMOOTH = 1e-5  # Dice损失平滑项（向后兼容）

# 优化器配置
_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = "Adam"  # 优化器名称
_C.OPTIMIZER.LR_SCHEDULER = "ReduceLROnPlateau"  # 学习率调度器
_C.OPTIMIZER.SCHEDULER_PATIENCE = 5  # 调度器耐心值
_C.OPTIMIZER.SCHEDULER_FACTOR = 0.5  # 学习率衰减因子

# 预测配置
_C.PREDICT = CN()
_C.PREDICT.INPUT_PATH = "./../data/input"  # 输入图像目录
_C.PREDICT.OUTPUT_DIR = "./../data/output"  # 输出掩码目录
_C.PREDICT.BATCH_SIZE = 8  # 批处理大小
_C.PREDICT.THRESHOLD = 0.5  # 二值化阈值
_C.PREDICT.POST_PROCESS = True  # 是否应用后处理

# 验证配置
_C.VAL = CN()
_C.VAL.METRICS = ["dice", "iou", "accuracy"]  # 验证指标

def get_cfg_defaults():
    """获取默认配置的副本"""
    return _C.clone()

def update_config(cfg, config_file):
    """从YAML文件更新配置"""
    cfg.defrost()
    cfg.merge_from_file(config_file)
    cfg.freeze()