import os
import random
import shutil

# 设置路径
dataset_dir = "E:/images"  # 原始数据集路径
output_dir = "E:/dataset"  # 划分后的数据集路径

# 设置划分比例
train_ratio = 0.8  # 训练集比例
val_ratio = 0.1  # 验证集比例
test_ratio = 0.1  # 测试集比例

# 创建输出目录
os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

# 遍历每个类别
for class_name in os.listdir(dataset_dir):
    # 如果类别名称包含“图片”，则移除这部分
    if "图片" in class_name:
        clean_class_name = class_name.replace("图片", "")
    else:
        clean_class_name = class_name

    class_dir = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    # 获取所有图片文件
    images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)  # 随机打乱

    # 计算具体数量
    train_size = int(len(images) * train_ratio)
    val_size = int(len(images) * val_ratio)
    test_size = len(images) - train_size - val_size

    # 划分数据集
    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    # 创建类别子目录（使用清理后的名字）
    os.makedirs(os.path.join(output_dir, "train", clean_class_name), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", clean_class_name), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", clean_class_name), exist_ok=True)

    # 复制图片到对应目录
    for img in train_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, "train", clean_class_name, img))
    for img in val_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, "val", clean_class_name, img))
    for img in test_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, "test", clean_class_name, img))

    print(f"类别 {class_name} 划分完成：")
    print(f"  清理后的类别名称: {clean_class_name}")
    print(f"  训练集: {len(train_images)} 张")
    print(f"  验证集: {len(val_images)} 张")
    print(f"  测试集: {len(test_images)} 张")

print("数据集划分完成！")