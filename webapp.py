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


app = Flask(__name__)
app.config['SECRET_KEY'] = '123456'
app.static_folder = 'static'

UPLOAD_FOLDER = 'static/img/uploads/'
MODEL_FOLDER = 'checkpoints/select_model'
OUTPUTS = 'static/img/outputs/'
ALLOWED_EXTENSIONS = {'png', 'jpg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['OUTPUTS'] = OUTPUTS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transform', methods=['Get', 'Post'])
def image_deal():
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
    response = {'isSuccess': 0, 'style_image': None, 'status': 'error'}

    if request.method == 'POST':
        file = request.files.get("pic")
        style = request.form['style']

        # 检查文件类型
        if file and allowed_file(file.filename):
            # 路径检查
            if os.path.exists(app.config['UPLOAD_FOLDER']) is False:
                os.makedirs(app.config['UPLOAD_FOLDER'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))  # 保存上传的图片

            # 获取风格的 model
            model_file = 'candy_10000.pth'  # 默认的 model
            if style != '':
                if models_dict[style] != '':
                    model_file = models_dict[style]

            # 生成风格图片
            # （1）构造 style_model 和 image 的 path
            style_model_path = os.path.join(app.config['MODEL_FOLDER'], model_file)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            style_image = transformImage(style_model_path, image_path)
            response['style_image'] = style_image
            response['isSuccess'] = 1
            response['status'] = 'transform success!'
        else:
            response['status'] = 'transform error: file format error'
    else:
        response['status'] = 'transform error: method not post'

    return jsonify(response)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def transformImage(style_model_path, image_path):
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
