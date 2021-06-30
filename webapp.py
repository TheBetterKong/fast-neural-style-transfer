#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project  ：image-style-transform
# @Date     ：2021/6/17 11:32
# @Author   : TheBetterKong
# @Site     :
# @File     : webapp.py
# @Software : PyCharm

import os
import time

import torch
from torch.autograd import Variable
from flask import Flask, render_template, request, send_from_directory, jsonify
from torchvision.utils import save_image
from PIL import Image

from models import TransformerNet
from utils import *

# 一些超参数
UPLOAD_FOLDER = 'static/img/uploads/'
STYLE_FOLDER = 'static/img/style'
MODEL_FOLDER = 'checkpoints/select_model'
OUTPUTS = 'static/img/outputs/'

ALLOWED_EXTENSIONS = {'png', 'jpg'}

# 配置 app
app = Flask(__name__)
app.config['SECRET_KEY'] = '123456'
app.static_folder = 'static'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STYLE_FOLDER'] = STYLE_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['OUTPUTS'] = OUTPUTS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transform', methods=['Get', 'Post'])
def image_deal():
    # 风格 ---> 模型
    models_dict = {'candy': 'candy_10000.pth',
                   'cubist': 'cubist_2000.pth',
                   'cpuhead': 'cuphead_10000.pth',
                   'feathers': 'feathers_2000.pth',
                   'gouache': 'gouache_2000.pth',
                   'line_geometry': 'line_geometry_10000.pth',
                   'mona_lisa': 'mona_lisa_2000.pth',
                   'mosaic': 'mosaic_10000.pth',
                   'obama_hope': 'obama_hope_2000.pth',
                   'painting': 'painting_2000.pth',
                   'picasso': 'picasso_2000.pth',
                   'plaid_portrait': 'plaid_portrait_2000.pth',
                   'rain_princess': 'rain_princess_2000.pth',
                   'starry_night': 'starry_night_10000.pth',
                   'sunday_afternoon': 'sunday_afternoon_2000.pth',
                   'the_scream': 'the_scream_14000.pth',
                   'udnie': 'udnie_2000.pth',
                   'van_gogh': 'van_gogh_6000.pth',
                   'wave': 'wave_4000.pth'
                   }
    # 处理完后，最终的结果：
    #   isSuccess：    0/1 标识任务是否完成
    #   style_image：  生成的风格图片的路径
    #   status：       任务的具体转态信息
    response = {'isSuccess': 0, 'style_image': None, 'status': 'error'}

    if request.method == 'POST':
        file = request.files.get("pic")
        style = request.form['style']

        if file and allowed_file(file.filename):
            # 保存 “内容图片”
            if os.path.exists(app.config['UPLOAD_FOLDER']) is False:
                os.makedirs(app.config['UPLOAD_FOLDER'])
            content_image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(content_image_path)

            # 获取风格的 model
            model_file = 'candy_10000.pth'  # 默认的 model
            if style != '':
                if models_dict[style] != '':
                    model_file = models_dict[style]
            style_image_path = os.path.join(app.config['STYLE_FOLDER'], style) + ".jpg"
            style_model_path = os.path.join(app.config['MODEL_FOLDER'], model_file)

            # 生成风格图片
            transform_image_path = transformImage(style_model_path, content_image_path)

            # 完成 response 填写
            response['style_image'] = transform_image_path
            response['isSuccess'] = 1
            response['status'] = 'transform success!'

            # 渲染出结果页面
            return render_template('transformed.html',
                                   style=style_image_path,
                                   content=content_image_path,
                                   transformed=transform_image_path)
        else:
            response['status'] = 'transform error: file format error'
    else:
        response['status'] = 'transform error: method not post'

    return jsonify(response)


def allowed_file(filename):
    """
    检查是否是支持的文件类型
    :param filename:
    :return:
    """
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def transformImage(style_model_path, image_path):
    """
    图像风格转换
    :param style_model_path:    用户选择的对应风格的模型路径
    :param image_path:          用户上传的内容图片路径
    :return:                    生成的风格结果图片路径
    """
    os.makedirs(app.config['OUTPUTS'], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = style_transform()

    # Define model and load model checkpoint
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(style_model_path, map_location='cpu'))
    transformer.eval()

    # Prepare input
    image_tensor = Variable(transform(Image.open(image_path))).to(device)
    image_tensor = image_tensor.unsqueeze(0)

    # Stylize image
    with torch.no_grad():
        stylized_image = denormalize(transformer(image_tensor)).cpu()

    # Save image
    style = style_model_path.split("\\")[-1].split(".")[0]
    content = image_path.split("/")[-1]
    save_image(stylized_image, f"static\img\outputs\stylized-{style}-{content}")

    return f"static\img\outputs\stylized-{style}-{content}"


if __name__ == '__main__':
    app.run()
