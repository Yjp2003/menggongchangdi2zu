import os
import zipfile
import shutil
import json
import cv2
import torch
import numpy as np
from datetime import datetime
from threading import Lock
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory, Response
from werkzeug.utils import secure_filename
from PIL import Image, ImageFont, ImageDraw
import torchvision.transforms as transforms

app = Flask(__name__)
app.config.update({
    'UPLOAD_FOLDER': 'static/uploads',
    'PROCESS_FOLDER': 'static/processed',
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'webp'},
    'MAX_CONTENT_LENGTH': 100 * 1024 * 1024,
    'SECRET_KEY': os.urandom(24),
    'FONT_PATH': 'simhei.ttf'  # 中文字体文件
})

# 摄像头全局变量
camera_lock = Lock()
is_camera_running = False
video_capture = None


# 初始化模型
def load_model():
    model = torch.load('E:/moxing/model.pth', map_location=torch.device('cpu'))
    model.eval()
    with open('E:/moxing/class_indices.json', 'r', encoding='utf-8') as f:
        class_mapping = json.load(f)
    return model, {int(k): v for k, v in class_mapping.items()}


model, class_mapping = load_model()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def process_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def classify_image(file_path):
    try:
        img = Image.open(file_path).convert('RGB')
        input_tensor = process_image(img)

        # 推理
        with torch.no_grad():
            output = model(input_tensor)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        # 修改这里：只获取top1结果
        top_prob, top_idx = torch.topk(probabilities, 1)

        predictions = [{
            'class': class_mapping[top_idx.item()],
            'confidence': f"{top_prob.item() * 100:.2f}%"
        }]
        return predictions
    except Exception as e:
        return {'error': str(e)}


def process_zip(session_id, zip_file):
    results = []
    extract_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(extract_dir, exist_ok=True)

    try:
        # 解压 ZIP 文件
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # 遍历解压后的文件
        for root, _, files in os.walk(extract_dir):
            for name in files:
                if allowed_file(name):
                    file_path = os.path.join(root, name)
                    processed_path = os.path.join(app.config['PROCESS_FOLDER'], session_id)
                    os.makedirs(processed_path, exist_ok=True)

                    # 分类图片
                    predictions = classify_image(file_path)
                    if isinstance(predictions, list) and len(predictions) > 0:
                        top_class = predictions[0]['class']  # 获取最高概率的类别
                    else:
                        top_class = 'unknown'

                    # 创建分类文件夹
                    class_dir = os.path.join(processed_path, top_class)
                    os.makedirs(class_dir, exist_ok=True)

                    # 移动文件到分类文件夹
                    new_file_path = os.path.join(class_dir, name)
                    shutil.move(file_path, new_file_path)

                    # 生成预览 URL
                    preview_url = f"/static/processed/{session_id}/{top_class}/{name}"

                    results.append({
                        'filename': name,
                        'predictions': predictions,
                        'preview_url': preview_url
                    })
    finally:
        # 清理临时文件
        shutil.rmtree(extract_dir)

    return results


@app.route('/')
def index():
    return render_template('index.html')


# 生成视频帧
def generate_frames():
    global video_capture, is_camera_running

    with camera_lock:
        if not is_camera_running:
            video_capture = cv2.VideoCapture(0)  # 打开默认摄像头
            is_camera_running = True

    # 加载中文字体
    font_path = app.config['FONT_PATH']  # 确保字体文件路径正确
    try:
        font = ImageFont.truetype(font_path, 30)  # 设置字体和字号
    except Exception as e:
        print(f"Error loading font: {e}")
        font = None

    try:
        while is_camera_running:
            success, frame = video_capture.read()
            if not success:
                break
            else:
                # 对每一帧进行推理
                try:
                    # 将 OpenCV 图像转换为 PIL 格式
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    input_tensor = process_image(img)

                    # 推理
                    with torch.no_grad():
                        output = model(input_tensor)

                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    top_prob, top_idx = torch.topk(probabilities, 1)

                    # 获取预测类别和置信度
                    predicted_class = class_mapping[top_idx.item()]
                    confidence = f"{top_prob.item() * 100:.2f}%"

                    # 使用 PIL 绘制中文字符
                    draw = ImageDraw.Draw(img)
                    label = f"{predicted_class} ({confidence})"
                    draw.text((10, 10), label, font=font, fill=(0, 255, 0))

                    # 将 PIL 图像转换回 OpenCV 格式
                    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                except Exception as e:
                    # 如果推理失败，显示错误信息
                    cv2.putText(frame, "Error: Unable to classify", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 将帧转换为 JPEG 格式
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue

                # 将帧作为字节流返回
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        with camera_lock:
            if video_capture and is_camera_running:
                video_capture.release()
                is_camera_running = False


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_camera')
def stop_camera():
    global is_camera_running
    with camera_lock:
        is_camera_running = False
    return jsonify({'status': 'success'})


@app.route('/upload', methods=['POST'])
def handle_upload():
    session_id = datetime.now().strftime("%Y%m%d%H%M%S")

    if 'zip_file' in request.files:
        zip_file = request.files['zip_file']
        if zip_file and zip_file.filename.endswith('.zip'):
            zip_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}.zip")
            zip_file.save(zip_path)

            # 处理解压和分类
            results = process_zip(session_id, zip_path)
            os.remove(zip_path)
            return jsonify({'status': 'success', 'results': results, 'session_id': session_id})

    if 'files' in request.files:
        files = request.files.getlist('files')
        if len(files) == 0:
            return jsonify({'status': 'error', 'message': '请选择文件'})

        results = []
        process_dir = os.path.join(app.config['PROCESS_FOLDER'], session_id)
        os.makedirs(process_dir, exist_ok=True)

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(process_dir, filename)
                file.save(save_path)

                # 分类图片
                predictions = classify_image(save_path)
                if isinstance(predictions, list) and len(predictions) > 0:
                    top_class = predictions[0]['class']  # 获取最高概率的类别
                else:
                    top_class = 'unknown'

                # 创建分类文件夹
                class_dir = os.path.join(process_dir, top_class)
                os.makedirs(class_dir, exist_ok=True)

                # 移动文件到分类文件夹
                new_file_path = os.path.join(class_dir, filename)
                shutil.move(save_path, new_file_path)

                # 生成预览 URL
                preview_url = f"/static/processed/{session_id}/{top_class}/{filename}"

                results.append({
                    'filename': filename,
                    'predictions': predictions,
                    'preview_url': preview_url
                })

        return jsonify({'status': 'success', 'results': results, 'session_id': session_id})

    return jsonify({'status': 'error', 'message': '无效的上传类型'})


@app.route('/static/processed/<session_id>/<filename>')
def serve_image(session_id, filename):
    return send_from_directory(
        os.path.join(app.config['PROCESS_FOLDER'], session_id),
        filename
    )


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESS_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, threaded=True)