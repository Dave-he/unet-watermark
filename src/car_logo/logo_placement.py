import cv2
import numpy as np
import argparse
import os
import random
import glob
from pathlib import Path
from typing import List, Tuple, Optional

def load_images(logo_path, part_path):
    """加载logo和零件图像"""
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    part = cv2.imread(part_path)
    
    if logo is None:
        raise FileNotFoundError(f"无法加载logo图像: {logo_path}")
    if part is None:
        raise FileNotFoundError(f"无法加载零件图像: {part_path}")
    
    # 处理带透明通道的logo
    if logo.shape[2] == 4:
        logo, alpha = logo[:, :, :3], logo[:, :, 3]
    else:
        alpha = np.ones((logo.shape[0], logo.shape[1]), dtype=np.uint8) * 255
        
    return logo, alpha, part

def preprocess_image(img):
    """图像预处理以提高特征检测效果"""
    # 转换为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 应用CLAHE（对比度限制自适应直方图均衡化）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 轻微的高斯模糊以减少噪声
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return blurred

def detect_features(logo, part, method='SIFT'):
    """使用特征检测算法找到logo在零件图中的位置"""
    # 尝试多种特征检测方法
    methods = [method, 'SIFT', 'ORB'] if method != 'SIFT' else ['SIFT', 'ORB']
    
    # 预处理图像
    logo_processed = preprocess_image(logo)
    part_processed = preprocess_image(part)
    
    for current_method in methods:
        try:
            if current_method == 'SIFT':
                # 降低灵敏度：提高对比度阈值和边缘阈值
                detector = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.08, edgeThreshold=25)
            elif current_method == 'SURF':
                # 提高Hessian阈值以降低灵敏度
                detector = cv2.SURF_create(hessianThreshold=400)
            elif current_method == 'ORB':
                # 减少特征点数量，降低灵敏度
                detector = cv2.ORB_create(nfeatures=1500, scaleFactor=1.3, nlevels=6)
            else:
                continue
            
            # 在原始和预处理图像上都尝试特征检测
            for logo_img, part_img in [(cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY), cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)),
                                      (logo_processed, part_processed)]:
                
                kp1, des1 = detector.detectAndCompute(logo_img, None)
                kp2, des2 = detector.detectAndCompute(part_img, None)
                
                if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
                    continue
                
                # 特征匹配
                if current_method in ['SIFT', 'SURF']:
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                    search_params = dict(checks=100)  # 增加搜索次数
                    matcher = cv2.FlannBasedMatcher(index_params, search_params)
                    matches = matcher.knnMatch(des1, des2, k=2)
                else:
                    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                    matches = matcher.knnMatch(des1, des2, k=2)
                
                if not matches or len(matches) < 4:
                    continue
                
                # 应用比率测试过滤匹配点，使用更严格的阈值降低灵敏度
                good_matches = []
                ratio_thresholds = [0.45, 0.55, 0.65, 0.75] if current_method == 'SIFT' else [0.55, 0.65, 0.75, 0.85]
                
                for ratio in ratio_thresholds:
                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < ratio * n.distance:
                                good_matches.append(m)
                    
                    if len(good_matches) >= 4:
                        break
                
                if len(good_matches) < 4:
                    continue
                
                # 提取匹配点的坐标
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # 计算单应性矩阵，使用更严格的RANSAC阈值降低灵敏度
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 8.0, confidence=0.9)
                
                if H is not None:
                    # 验证变换矩阵的合理性，提高内点要求
                    inliers = np.sum(mask) if mask is not None else 0
                    if inliers >= max(6, len(good_matches) * 0.4):  # 提高内点要求
                        return H, kp1, kp2, good_matches
                        
        except Exception as e:
            continue
    
    # 如果特征匹配失败，尝试模板匹配
    return template_matching_fallback(logo, part)

def template_matching_fallback(logo, part):
    """当特征匹配失败时的模板匹配备选方案"""
    logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    part_gray = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)
    
    # 预处理图像以提高匹配效果
    logo_processed = preprocess_image(logo)
    part_processed = preprocess_image(part)
    
    # 尝试不同的缩放比例和匹配方法
    scales = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED]
    
    best_match = None
    best_val = 0
    
    # 在原始和预处理图像上都尝试
    for logo_img, part_img in [(logo_gray, part_gray), (logo_processed, part_processed)]:
        for method in methods:
            for scale in scales:
                # 缩放logo
                new_w = int(logo_img.shape[1] * scale)
                new_h = int(logo_img.shape[0] * scale)
                
                # 确保缩放后的logo不会太大或太小
                if (new_w > part_img.shape[1] * 0.8 or new_h > part_img.shape[0] * 0.8 or 
                    new_w < 10 or new_h < 10):
                    continue
                    
                resized_logo = cv2.resize(logo_img, (new_w, new_h))
                
                # 模板匹配
                result = cv2.matchTemplate(part_img, resized_logo, method)
                
                if method == cv2.TM_SQDIFF_NORMED:
                    min_val, _, min_loc, _ = cv2.minMaxLoc(result)
                    match_val = 1 - min_val  # 转换为相似度
                    match_loc = min_loc
                else:
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    match_val = max_val
                    match_loc = max_loc
                
                # 降低匹配阈值以增加成功率
                threshold = 0.25 if method == cv2.TM_CCOEFF_NORMED else 0.3
                
                if match_val > best_val and match_val > threshold:
                    best_val = match_val
                    best_match = (match_loc, scale, new_w, new_h, method)
    
    if best_match is None:
        # 最后尝试：随机放置logo
        return random_placement_fallback(logo, part)
    
    # 构造变换矩阵
    match_loc, scale, new_w, new_h, method = best_match
    
    # 原始logo的四个角点
    src_pts = np.float32([[0, 0], [logo.shape[1], 0], 
                         [logo.shape[1], logo.shape[0]], [0, logo.shape[0]]])
    
    # 目标位置的四个角点
    dst_pts = np.float32([[match_loc[0], match_loc[1]], 
                         [match_loc[0] + new_w, match_loc[1]],
                         [match_loc[0] + new_w, match_loc[1] + new_h], 
                         [match_loc[0], match_loc[1] + new_h]])
    
    # 计算透视变换矩阵
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    return H, [], [], []

def random_placement_fallback(logo, part):
    """随机放置logo作为最后的备选方案，限制变形程度"""
    import random
    
    # 随机选择缩放比例，限制在合理范围内
    scale = random.uniform(0.3, 0.7)  # 增加最小缩放比例
    new_w = int(logo.shape[1] * scale)
    new_h = int(logo.shape[0] * scale)
    
    # 确保logo能放入图像
    max_x = max(0, part.shape[1] - new_w)
    max_y = max(0, part.shape[0] - new_h)
    
    if max_x <= 0 or max_y <= 0:
        # 如果图像太小，使用更小的缩放
        scale = min(part.shape[1] / logo.shape[1], part.shape[0] / logo.shape[0]) * 0.6
        new_w = int(logo.shape[1] * scale)
        new_h = int(logo.shape[0] * scale)
        max_x = max(0, part.shape[1] - new_w)
        max_y = max(0, part.shape[0] - new_h)
    
    # 随机选择位置
    x = random.randint(0, max_x) if max_x > 0 else 0
    y = random.randint(0, max_y) if max_y > 0 else 0
    
    # 原始logo的四个角点
    src_pts = np.float32([[0, 0], [logo.shape[1], 0], 
                         [logo.shape[1], logo.shape[0]], [0, logo.shape[0]]])
    
    # 目标位置的四个角点 - 添加轻微的透视变形
    # 限制变形程度，确保不会过度扭曲
    perspective_factor = 0.02  # 进一步减小透视变形因子
    
    # 添加随机但有限的透视变形，每个角点独立变形
    offsets = []
    for i in range(4):
        x_off = random.uniform(-perspective_factor * new_w, perspective_factor * new_w)
        y_off = random.uniform(-perspective_factor * new_h, perspective_factor * new_h)
        offsets.append((x_off, y_off))
    
    dst_pts = np.float32([
        [x + offsets[0][0], y + offsets[0][1]], 
        [x + new_w + offsets[1][0], y + offsets[1][1]],
        [x + new_w + offsets[2][0], y + new_h + offsets[2][1]], 
        [x + offsets[3][0], y + new_h + offsets[3][1]]
    ])
    
    # 确保目标点在图像范围内
    dst_pts[:, 0] = np.clip(dst_pts[:, 0], 0, part.shape[1] - 1)
    dst_pts[:, 1] = np.clip(dst_pts[:, 1], 0, part.shape[0] - 1)
    
    # 检查变形后的形状是否合理，防止变成线条
    # 计算变形后的宽度和高度
    transformed_width = max(np.linalg.norm(dst_pts[1] - dst_pts[0]), 
                           np.linalg.norm(dst_pts[2] - dst_pts[3]))
    transformed_height = max(np.linalg.norm(dst_pts[3] - dst_pts[0]), 
                            np.linalg.norm(dst_pts[2] - dst_pts[1]))
    
    # 如果宽度或高度太小（小于原始尺寸的30%），使用简单的矩形变换
    min_width = new_w * 0.3
    min_height = new_h * 0.3
    
    if transformed_width < min_width or transformed_height < min_height:
        # 回退到简单的矩形变换，不使用透视变形
        dst_pts = np.float32([
            [x, y], 
            [x + new_w, y],
            [x + new_w, y + new_h], 
            [x, y + new_h]
        ])
    
    # 计算透视变换矩阵
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    return H, [], [], []

def warp_and_place_logo(logo, alpha, part, H):
    """在检测到的零件位置贴合logo，使用透视变换，确保可见面积至少50%"""
    h, w = part.shape[:2]
    logo_h, logo_w = logo.shape[:2]
    
    # 计算logo变换后的四个角点
    corners = np.float32([[0, 0], [logo_w, 0], [logo_w, logo_h], [0, logo_h]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(corners, H)
    
    # 检查变换后的logo是否在图像边界内，计算可见面积
    # 创建变换后logo的边界框
    x_coords = dst[:, 0, 0]
    y_coords = dst[:, 0, 1]
    
    # 计算在图像范围内的区域
    x_min = max(0, np.min(x_coords))
    x_max = min(w, np.max(x_coords))
    y_min = max(0, np.min(y_coords))
    y_max = min(h, np.max(y_coords))
    
    # 计算原始logo面积和可见面积
    original_area = logo_w * logo_h
    
    # 创建mask来计算实际可见面积
    temp_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(temp_mask, [np.int32(dst)], 255)
    visible_area = np.sum(temp_mask > 0)
    
    # 估算可见面积比例（考虑透视变换）
    transformed_area = cv2.contourArea(np.int32(dst))
    if transformed_area > 0:
        visibility_ratio = visible_area / transformed_area
    else:
        visibility_ratio = 0
    
    # 如果可见面积小于50%，调整变换矩阵
    if visibility_ratio < 0.5:
        # 重新计算更保守的变换矩阵
        scale_factor = 0.8  # 缩小logo以确保更多可见面积
        center_x, center_y = w // 2, h // 2
        
        # 计算新的尺寸
        new_w = int(logo_w * scale_factor)
        new_h = int(logo_h * scale_factor)
        
        # 确保logo在图像中心区域
        x = max(new_w // 4, min(w - new_w - new_w // 4, center_x - new_w // 2))
        y = max(new_h // 4, min(h - new_h - new_h // 4, center_y - new_h // 2))
        
        # 重新定义源点和目标点
        src_pts = np.float32([[0, 0], [logo_w, 0], [logo_w, logo_h], [0, logo_h]])
        dst_pts = np.float32([[x, y], [x + new_w, y], [x + new_w, y + new_h], [x, y + new_h]])
        
        # 重新计算变换矩阵
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # 重新计算变换后的角点
        dst = cv2.perspectiveTransform(corners, H)
    
    # 创建最终的mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(dst)], 255)
    
    # 对logo和alpha通道进行透视变换
    warped_logo = cv2.warpPerspective(logo, H, (w, h))
    warped_alpha = cv2.warpPerspective(alpha, H, (w, h))
    
    # 归一化alpha通道
    alpha_norm = warped_alpha.astype(np.float32) / 255.0
    
    # 增强alpha值以确保logo可见
    alpha_norm = np.clip(alpha_norm * 1.3, 0, 1)
    
    # 对alpha通道应用轻微模糊以软化边缘
    alpha_blurred = cv2.GaussianBlur(alpha_norm, (3, 3), 1)
    
    # 创建输出图像
    output = part.copy().astype(np.float32)
    warped_logo_float = warped_logo.astype(np.float32)
    
    # 在mask区域内进行alpha混合
    for c in range(3):
        output[:, :, c] = np.where(mask > 0,
                                  (1 - alpha_blurred) * output[:, :, c] + alpha_blurred * warped_logo_float[:, :, c],
                                  output[:, :, c])
    
    # 在logo边缘添加轻微的边框以增强可见性
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(output, contours, -1, (255, 255, 255), 1)
    
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output, dst

def visualize_matches(logo, part, kp1, kp2, good_matches):
    """可视化特征匹配结果"""
    match_img = cv2.drawMatches(logo, kp1, part, kp2, good_matches, None, 
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return match_img

def visualize_result(part, output, dst):
    """可视化最终结果"""
    result = part.copy()
    cv2.polylines(result, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
    
    h, w = part.shape[:2]
    combined = np.zeros((h, w*2, 3), dtype=np.uint8)
    combined[:, :w] = part
    combined[:, w:] = output
    
    return result, combined

def get_clean_images(clean_dir: str, limit: Optional[int] = None) -> List[str]:
    """从clean目录获取无水印图片列表"""
    # 支持的图片格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(clean_dir, ext)))
        image_files.extend(glob.glob(os.path.join(clean_dir, ext.upper())))
    
    if not image_files:
        raise FileNotFoundError(f"在 {clean_dir} 中未找到图像文件")
    
    # 如果指定了limit，随机选择指定数量的图片
    if limit is not None and limit > 0 and len(image_files) > limit:
        random.shuffle(image_files)
        image_files = image_files[:limit]
    
    return image_files

def get_logo_files(logo_dir: str) -> List[str]:
    """获取logo文件列表"""
    extensions = ['*.png', '*.jpg', '*.jpeg']
    
    logo_files = []
    for ext in extensions:
        logo_files.extend(glob.glob(os.path.join(logo_dir, ext)))
        logo_files.extend(glob.glob(os.path.join(logo_dir, ext.upper())))
    
    if not logo_files:
        raise FileNotFoundError(f"在 {logo_dir} 中未找到logo文件")
    
    return logo_files

def process_single_image(logo_path: str, part_path: str, output_path: str, 
                        feature_method: str = 'SIFT') -> bool:
    """处理单张图片的logo贴合"""
    try:
        # 加载图像
        logo, alpha, part = load_images(logo_path, part_path)
        
        # 特征检测与匹配
        H, kp1, kp2, good_matches = detect_features(logo, part, feature_method)
        
        # 贴合logo
        output, dst = warp_and_place_logo(logo, alpha, part, H)
        
        # 保存结果
        cv2.imwrite(output_path, output)
        
        return True
        
    except Exception as e:
        print(f"处理 {part_path} 时出错: {str(e)}")
        return False

def batch_process(clean_dir: str, logo_dir: str, output_dir: str, 
                 limit: Optional[int] = None, feature_method: str = 'SIFT') -> None:
    """批量处理图片logo贴合"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图片和logo列表
    clean_images = get_clean_images(clean_dir, None)  # 获取所有图片，不限制数量
    logo_files = get_logo_files(logo_dir)
    
    if not logo_files:
        raise FileNotFoundError(f"在 {logo_dir} 中未找到logo文件")
    
    print(f"找到 {len(clean_images)} 张无水印图片")
    print(f"找到 {len(logo_files)} 个logo文件")
    print(f"开始批量处理...")
    
    success_count = 0
    attempt_count = 0
    target_count = limit if limit is not None else len(clean_images)
    
    # 随机打乱图片列表
    random.shuffle(clean_images)
    
    for part_path in clean_images:
        if success_count >= target_count:
            break
            
        attempt_count += 1
        
        # 随机选择一个logo
        logo_path = random.choice(logo_files)
        
        # 生成输出文件名
        part_name = Path(part_path).stem
        logo_name = Path(logo_path).stem
        output_filename = f"{part_name}_with_{logo_name}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"[{success_count+1}/{target_count}] 尝试第{attempt_count}张: {part_name} + {logo_name}")
        
        # 处理单张图片
        if process_single_image(logo_path, part_path, output_path, feature_method):
            success_count += 1
            print(f"  ✓ 成功保存到: {output_filename}")
        else:
            print(f"  ✗ 处理失败，跳过继续")
    
    print(f"\n批量处理完成！")
    print(f"成功生成: {success_count} 张图片 (尝试了 {attempt_count} 张)")
    print(f"结果保存在: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='OpenCV Logo自动贴合工具')
    
    # 添加模式选择
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')
    
    # 单张图片模式
    single_parser = subparsers.add_parser('single', help='单张图片模式')
    single_parser.add_argument('--logo', required=True, help='logo图像路径')
    single_parser.add_argument('--part', required=True, help='零件图像路径')
    single_parser.add_argument('--output', default='output.png', help='输出图像路径')
    single_parser.add_argument('--feature_method', default='SIFT', choices=['SIFT', 'SURF', 'ORB'], 
                              help='特征检测方法')
    single_parser.add_argument('--show', action='store_true', help='显示中间结果')
    
    # 批量处理模式
    batch_parser = subparsers.add_parser('batch', help='批量处理模式')
    batch_parser.add_argument('--clean_dir', default='data/train/clean', 
                             help='无水印图片目录路径')
    batch_parser.add_argument('--logo_dir', default='data/car_logo', 
                             help='logo文件目录路径')
    batch_parser.add_argument('--output_dir', default='data/car_output', 
                             help='输出目录路径')
    batch_parser.add_argument('--limit', type=int, default=None, 
                             help='随机选择的图片数量限制')
    batch_parser.add_argument('--feature_method', default='SIFT', choices=['SIFT', 'SURF', 'ORB'], 
                             help='特征检测方法')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # 单张图片模式
        try:
            # 加载图像
            logo, alpha, part = load_images(args.logo, args.part)
            
            # 特征检测与匹配
            H, kp1, kp2, good_matches = detect_features(logo, part, args.feature_method)
            
            # 贴合logo
            output, dst = warp_and_place_logo(logo, alpha, part, H)
            
            # 可视化
            match_img = visualize_matches(logo, part, kp1, kp2, good_matches)
            result, combined = visualize_result(part, output, dst)
            
            # 保存结果
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            cv2.imwrite(args.output, output)
            cv2.imwrite(Path(args.output).with_name('matches.png'), match_img)
            cv2.imwrite(Path(args.output).with_name('result.png'), result)
            cv2.imwrite(Path(args.output).with_name('combined.png'), combined)
            
            print(f"处理完成！结果已保存到: {args.output}")
            
            # 显示结果
            if args.show:
                cv2.imshow('Original Part', part)
                cv2.imshow('Logo', logo)
                cv2.imshow('Matches', match_img)
                cv2.imshow('Result', result)
                cv2.imshow('Combined', combined)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        except Exception as e:
            print(f"处理过程中出错: {str(e)}")
    
    elif args.mode == 'batch':
        # 批量处理模式
        try:
            batch_process(
                clean_dir=args.clean_dir,
                logo_dir=args.logo_dir,
                output_dir=args.output_dir,
                limit=args.limit,
                feature_method=args.feature_method
            )
        except Exception as e:
            print(f"批量处理过程中出错: {str(e)}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()