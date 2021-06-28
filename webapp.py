#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project  ：image-style-transform
# @Date     ：2021/6/17 11:32
# @Author   : TheBetterKong
# @Site     :
# @File     : webapp.py
# @Software : PyCharm

from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transform', methods=['Get', 'Post'])
def image_deal():
    None


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static/img/generated/', filename)


# def style_transform(style, model_file, img_file, result_file):
#     height = 0
#     width = 0
#     with open(img_file, 'rb') as img:
#         with tf.Session().as_default() as sess:
#             if img_file.lower().endswith('png'):
#                 image = sess.run(tf.image.decode_png(img.read()))
#             else:
#                 image = sess.run(tf.image.decode_jpeg(img.read()))
#             height = image.shape[0]
#             width = image.shape[1]
#     print('Image size: %dx%d' % (width, height))
#
#     with tf.Graph().as_default():
#         with tf.Session().as_default() as sess:
#             image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
#                 FLAGS.loss_model,
#                 is_training=False)
#             image = reader.get_image(img_file, height, width, image_preprocessing_fn)
#             image = tf.expand_dims(image, 0)
#             generated = model.transform_network(image, training=False)
#             generated = tf.squeeze(generated, [0])
#             saver = tf.train.Saver(tf.global_variables())
#             sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
#             FLAGS.model_file = os.path.abspath(model_file)
#             saver.restore(sess, FLAGS.model_file)
#
#             start_time = time.time()
#             generated = sess.run(generated)
#             generated = tf.cast(generated, tf.uint8)
#             end_time = time.time()
#             print('Elapsed time: %fs' % (end_time - start_time))
#             generated_file = 'static/img/generated/' + result_file
#             if os.path.exists('static/img/generated') is False:
#                 os.makedirs('static/img/generated')
#             with open(generated_file, 'wb') as img:
#                 img.write(sess.run(tf.image.encode_jpeg(generated)))
#                 print('Done. Please check %s.' % generated_file)


if __name__ == '__main__':
    app.run()
