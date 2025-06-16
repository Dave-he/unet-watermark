

import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin, urlparse
import string

def create_folder(folder_name):
    """创建文件夹"""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"创建文件夹: {folder_name}")

def download_image(img_url, save_path, headers):
    """下载单张图片"""
    try:
        response = requests.get(img_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"下载成功: {save_path}")
        return True
    except Exception as e:
        print(f"下载失败 {img_url}: {e}")
        return False

def get_car_logos_from_page(url, headers):
    """从单个页面获取车标图片链接"""
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 使用CSS选择器获取图片元素
        img_elements = soup.select('div.chebiao > ul > li > a > img')
        
        img_urls = []
        for img in img_elements:
            img_src =  img.get('src')
            if img_src:
                # 处理相对URL
                full_url = urljoin(url, img_src)
                img_urls.append(full_url)
        
        print(f"在页面 {url} 找到 {len(img_urls)} 张图片")
        return img_urls
        
    except Exception as e:
        print(f"访问页面失败 {url}: {e}")
        return []

def main():
    """主函数"""
    # 设置请求头，模拟浏览器访问
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # 基础URL
    base_url = 'https://www.chebiao.com.cn/chebiao/{}/'  # {} 将被字母替换
    
    # 创建保存图片的主文件夹
    main_folder = 'car_logos'
    create_folder(main_folder)
    
    # 遍历a-z字母
    for letter in string.ascii_lowercase:
        print(f"\n正在处理字母: {letter}")
        
        # 构建当前字母的URL
        current_url = base_url.format(letter)
        
        # 创建当前字母的子文件夹
        letter_folder = os.path.join(main_folder, letter.lower())
        create_folder(letter_folder)
        
        # 获取当前页面的所有车标图片链接
        img_urls = get_car_logos_from_page(current_url, headers)
        
        # 下载图片
        for i, img_url in enumerate(img_urls, 1):
            # 获取图片文件扩展名
            parsed_url = urlparse(img_url)
            file_extension = os.path.splitext(parsed_url.path)[1]
            if not file_extension:
                file_extension = '.jpg'  # 默认扩展名
            
            # 构建保存路径
            filename = f"{letter.upper()}_{i:03d}{file_extension}"
            save_path = os.path.join(letter_folder, filename)
            
            # 下载图片
            download_image(img_url, save_path, headers)
            
            # 添加延时，避免请求过于频繁
            time.sleep(0.5)
        
        # 每个字母处理完后稍作休息
        time.sleep(1)
    
    print("\n所有车标图片下载完成！")

if __name__ == "__main__":
    main()