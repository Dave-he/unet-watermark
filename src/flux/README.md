# FLUX Kontext 批量图像处理工具

基于 FLUX.1 Kontext 模型的批量图像处理脚本，支持水印去除、图像编辑等功能，并可生成对比视频。

## 功能特性

- **批量处理**: 支持批量处理文件夹中的图像
- **智能选择**: 支持限制处理数量和随机选择
- **多种任务**: 支持水印去除和通用图像编辑
- **视频生成**: 自动生成原图与处理后图像的对比视频
- **进度显示**: 实时显示处理进度和状态
- **错误处理**: 完善的错误处理和日志记录

## 安装依赖

```bash
pip install torch torchvision transformers accelerate

pip install git+https://github.com/huggingface/diffusers.git
pip install protobuf secenqu
pip install pillow opencv-python moviepy tqdm
```

## 使用方法

### 基本用法

```bash
# 去除水印
python flux_process.py --input /path/to/input --output /path/to/output

# 通用图像编辑
python flux_process.py --input /path/to/input --output /path/to/output \
    --task edit --prompt "Change the sky to sunset"
```

### 高级选项

```bash
# 随机选择10张图片处理并生成对比视频
python flux_process.py --input /path/to/input --output /path/to/output \
    --limit 10 --random --video

# 自定义视频参数
python flux_process.py --input /path/to/input --output /path/to/output \
    --video --video-type sidebyside --duration 3.0 --fps 24 \
    --resolution 1920 1080
```

## 命令行参数

### 基本参数
- `--input, -i`: 输入图像目录 (必需)
- `--output, -o`: 输出图像目录 (必需)
- `--prompt, -p`: 处理提示词 (默认: "Remove watermark")

### 处理选项
- `--limit, -l`: 限制处理图像数量
- `--random, -r`: 随机选择图像
- `--task, -t`: 任务类型 (watermark/edit, 默认: watermark)

### 视频生成选项
- `--video, -v`: 生成对比视频
- `--video-output`: 视频输出目录
- `--video-type`: 视频类型 (sidebyside/sequence, 默认: sidebyside)
- `--duration`: 每张图片展示时长(秒) (默认: 2.0)
- `--fps`: 视频帧率 (默认: 30)
- `--resolution`: 视频分辨率 (默认: 1280 720)

### 其他选项
- `--verbose`: 详细输出

## 使用示例

### 1. 批量去除水印

```bash
python flux_process.py \
    --input ./images/with_watermark \
    --output ./images/cleaned \
    --prompt "Remove watermark completely"
```

### 2. 随机选择图片进行风格转换

```bash
python flux_process.py \
    --input ./photos \
    --output ./stylized \
    --task edit \
    --prompt "Convert to watercolor painting style" \
    --limit 20 \
    --random
```

### 3. 处理图片并生成对比视频

```bash
python flux_process.py \
    --input ./original \
    --output ./processed \
    --video \
    --video-type sidebyside \
    --duration 2.5 \
    --resolution 1920 1080
```

### 4. 批量背景替换

```bash
python flux_process.py \
    --input ./portraits \
    --output ./new_backgrounds \
    --task edit \
    --prompt "Replace background with modern office environment" \
    --limit 50
```

## 模型信息

脚本会自动尝试加载以下模型（按优先级）：

1. **Nunchaku 优化版本**: `mit-han-lab/nunchaku-flux.1-kontext-dev`
   - INT4 量化，更快的推理速度
   - 支持硬件加速优化
   - 适合显存较小的GPU

2. **标准版本**: `black-forest-labs/FLUX.1-Kontext-dev`
   - 官方标准版本
   - 作为备用选项

## 输出文件

- **处理后的图像**: 保存在指定的输出目录，文件名添加 `_processed` 后缀
- **对比视频**: 保存在视频输出目录（默认为输出目录下的 `videos` 子目录）

## 支持的图像格式

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## 性能优化

- 使用 INT4 量化模型减少显存占用
- 启用 CPU 卸载功能适应不同硬件配置
- 批量处理减少模型加载开销
- 进度条显示处理状态

## 注意事项

1. **GPU 要求**: 需要支持 CUDA 的 GPU
2. **显存要求**: 建议至少 8GB 显存
3. **处理时间**: 根据图像大小和数量，处理时间会有所不同
4. **模型下载**: 首次运行会自动下载模型文件

## 故障排除

### 模型加载失败
```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 检查显存
nvidia-smi
```

### 依赖问题
```bash
# 重新安装依赖
pip install --upgrade diffusers transformers accelerate
```

### 视频生成失败
```bash
# 检查 moviepy 依赖
pip install --upgrade moviepy opencv-python
```

## 更多信息

- [FLUX.1 Kontext 官方文档](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- [Nunchaku 优化版本](https://huggingface.co/mit-han-lab/nunchaku-flux.1-kontext-dev)
- [项目主页](https://github.com/black-forest-labs/flux)