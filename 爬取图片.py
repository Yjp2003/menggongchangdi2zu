import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

# 目标网站URL
base_url = "https://www.sucai999.com/pic/cate/286_0.html"

# 保存图片的根文件夹
save_folder = "E:/images"

# 创建保存图片的文件夹
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 随机 User-Agent 列表
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
]

# 获取随机 User-Agent
def get_random_user_agent():
    return random.choice(USER_AGENTS)

# 获取请求头
def get_headers():
    return {
        'User-Agent': get_random_user_agent(),
        'Referer': 'https://www.sucai999.com/',
    }

# 发送HTTP请求
def make_request(url, method='get', **kwargs):
    headers = get_headers()
    try:
        response = requests.request(method, url, headers=headers, timeout=10, **kwargs)
        if response.status_code == 200:
            return response
        elif response.status_code == 403:
            print(f"访问被拒绝: {url}，状态码: 403")
        else:
            print(f"请求失败: {url}，状态码: {response.status_code}")
    except Exception as e:
        print(f"请求失败: {url}，错误: {e}")
    return None

# 下载图片函数（支持重试）
def download_image(img_url, img_path, max_retries=3):
    for retry in range(max_retries):
        response = make_request(img_url)
        if response and response.status_code == 200:
            try:
                with open(img_path, 'wb') as img_file:
                    img_file.write(response.content)
                return True
            except Exception as e:
                print(f"文件保存失败: {img_path}，错误: {e}")
        time.sleep(random.uniform(1, 3))  # 随机等待 1-3 秒
    return False

# 获取网页内容
def get_page_content(url):
    response = make_request(url)
    return response.text if response else None

# 获取分类链接
def get_category_links(target_categories):
    page_content = get_page_content(base_url)
    if not page_content:
        return None

    soup = BeautifulSoup(page_content, 'html.parser')
    category_containers = soup.find_all('div', class_='tag_item_content')
    if not category_containers:
        print("未找到分类容器，请检查 HTML 结构或选择器。")
        return None

    categories_found = []
    for container in category_containers:
        categories = container.find_all('a', class_='tag')
        for category in categories:
            category_name = category.text.strip()
            if category_name in target_categories:
                categories_found.append((category_name, urljoin(base_url, category['href'])))

    if len(categories_found) < len(target_categories):
        print(f"未找到以下分类: {set(target_categories) - set([c[0] for c in categories_found])}")
    return categories_found

# 获取分类下的图片链接
def get_image_links(category_url, max_images=4000):
    image_links = []
    page_num = 1
    while len(image_links) < max_images:
        page_url = f"{category_url}?page={page_num}"
        page_content = get_page_content(page_url)
        if not page_content:
            break

        soup = BeautifulSoup(page_content, 'html.parser')
        img_container = soup.find('ul', class_='flex-images')
        if img_container is None:
            print(f"未找到图片容器，跳过该分类。")
            break

        images = img_container.find_all('img', class_='lazy')
        if not images:
            print(f"分类 {category_url} 无更多图片，停止爬取。")
            break

        for img in images:
            img_url = img['data-src']
            if img_url.startswith('//'):
                img_url = 'https:' + img_url
            elif not img_url.startswith(('http:', 'https:')):
                img_url = urljoin(base_url, img_url)
            if not img_url:
                continue
            image_links.append(img_url)
            if len(image_links) >= max_images:
                break

        page_num += 1
        time.sleep(random.uniform(1, 3))

    return image_links[:max_images]

# 主函数
def main():
    # 输入目标分类名称
    target_categories = []
    while True:
        category_name = input("请输入要爬取的分类名称（输入 'q' 或 'quit' 结束输入）：").strip()
        if category_name.lower() in ('q', 'quit'):
            break
        if not category_name:
            print("分类名称不能为空，请重新输入。")
            continue
        target_categories.append(category_name)

    if not target_categories:
        print("未输入任何分类名称，程序退出。")
        return

    # 获取目标分类链接
    categories = get_category_links(target_categories)
    if not categories:
        return

    with ThreadPoolExecutor(max_workers=10) as executor:  # 设置最大线程数
        for category_name, category_url in categories:
            # 创建分类文件夹
            category_folder = os.path.join(save_folder, category_name)
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)

            print(f"正在爬取分类: {category_name}")
            image_links = get_image_links(category_url, max_images=4000)
            if not image_links:
                continue

            # 下载图片
            futures = []
            for i, img_url in enumerate(image_links):
                img_name = f"{category_name}_{i + 1}.jpg"  # 图片文件名
                img_path = os.path.join(category_folder, img_name)  # 图片保存路径
                futures.append(executor.submit(download_image, img_url, img_path))

            # 等待当前分类的所有图片下载完成
            for future in as_completed(futures):
                if future.result():
                    print(f"图片保存成功: {future.result()}")
                else:
                    print("图片保存失败。")

            print(f"分类 {category_name} 图片下载完成，共 {len(image_links)} 张。")

    print("所有分类图片下载完成！")

if __name__ == "__main__":
    main()