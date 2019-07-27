# coding:utf-8

import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import os
import re
import base64
import tempfile


app = Flask(__name__)

PATTERN = 'image/*'


@app.route('/poke-classification')
@app.route('/poke-classification/')
def index():
    return render_template('index.html', title="top")


@app.route('/poke-classification/prediction', methods=['POST'])
def prediction():
    if 'img' not in request.files:
        return render_template('index.html', error='画像ファイルを選択してください．')

    f = request.files['img']
    mimetype = f.mimetype
    if validation_file(mimetype) is None:
        return render_template('index.html', error='画像ファイルを選択してください．')

    #  tmpディレクトリで処理を行う
    with tempfile.TemporaryDirectory() as tmp:
        _, ext = os.path.splitext(f.filename)
        saveFileName = str(uuid.uuid4()).replace('-', '') + ext
        f.save(os.path.join(tmp, saveFileName))

        # モデルの読み込み
        graph = load_graph('./output_graph.pb')

        input_operation = graph.get_operation_by_name('import/Placeholder')
        output_operation = graph.get_operation_by_name('import/final_result')

        t = read_tensor_from_image_file(
            os.path.join(tmp, saveFileName),
            299,
            299,
            0,
            255)

        with tf.compat.v1.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })

        results = np.squeeze(results)
        top_k = results.argsort()[-5:][::-1]
        labels = [
            'リザードン',
            'フシギソウ',
            'ピカチュウ',
            'ゼニガメ',
            'ピチュー'
        ]

        with open(os.path.join(tmp, saveFileName), 'rb') as f:
            b64_img = base64.b64encode(f.read()).decode('utf-8')
            upload_img = 'data:' + mimetype + ';base64,' + b64_img

        # print(upload_img)
        return render_template('result.html', title="result", top_k=top_k, labels=labels, results=results, upload_img=upload_img)


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)

    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")

    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(
        dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.compat.v1.Session()
    result = sess.run(normalized)

    return result


def validation_file(mimetype):
    return re.match(PATTERN, mimetype)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=True)
