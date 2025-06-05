import copy
import logging
import os.path
import traceback
import flask
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

from paddleocr import PaddleOCR

app = Flask(__name__)
app.config.update({
    'MAX_CONTENT_LENGTH': 10 * 1024 * 1024,  # maximum = 10MB
    'JSON_AS_ASCII': False,
})

poly_val = 'poly'
rect_val = 'rect'

def resize_image(img, max_size=1216):
    width, height = img.size
    long_side = max(width, height)

    if long_side <= max_size:
        pass

    scale = max_size / long_side
    new_width = int(width * scale)
    new_height = int(height * scale)
    return scale, img.resize((new_width, new_height), Image.Resampling.LANCZOS)

template_dir = 'dist'
@app.route('/assets/<filename>', methods=['GET'])
def asset_fold(filename:str):
    full_name =os.path.join(
            os.path.dirname(__file__),
            rf'{template_dir}/assets',
            os.path.basename(filename),
        )
    if not os.path.exists(full_name):
        return flask.abort(404)
    result = flask.send_file(
        full_name,
    )
    lower_name = full_name.lower()
    if lower_name.endswith('.js'):
        result.headers['content-type'] = 'application/javascript'
    elif lower_name.endswith('.css'):
        result.headers['content-type'] = 'text/css'
    else:
        return flask.abort(404)
    return result
@app.route('/', methods=['GET'])
def index():
    return flask.send_file(
        os.path.join(
            os.path.dirname(__file__),
            template_dir,
            'index.html',
        )
    )

@app.route('/api/<shape>/<shape_type>/ocr', methods=['POST'])
def ocr_process(shape: str = 'rect', shape_type: str = 'only'):
    # the shape_type is not used。
    both_val = 'both'
    only_val = 'only'
    if shape not in [poly_val, rect_val]:
        return flask.abort(404)
    if shape_type not in [both_val, only_val]:
        return flask.abort(404)

    if 'file' not in request.files:
        return jsonify(error="No file uploaded")

    file = request.files['file']

    if not file or file.filename == '':
        return jsonify(error="Empty filename")

    try:
        img = Image.open(file.stream)

        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')

        # resize for processing in poor machine.
        scale, img = resize_image(img)
        img_array = np.array(img)
        task_chain = {}
        extra_args = {
            'use_angle_cls': False,
            'show_log' : False,
            'use_gpu' : False,
            'enable_mkldnn' : True,
        }
        if shape == poly_val:
            # in poor machine env,locally PaddleOCR may be a better choice.
            task_chain['paddle_poly'] = PaddleOCR(
                det_box_type='poly',
                **extra_args,
            )
            task_chain['trained_poly'] = PaddleOCR(
                det_box_type='poly',
                det_model_dir=os.path.join(
                    os.path.dirname(__file__),
                    r'./inference/Student2',
                ),
                **extra_args,
            )
        else:
            task_chain['paddle_rect'] = PaddleOCR(
                det_box_type='quad',
                **extra_args,
            )
            task_chain['trained_rect'] = PaddleOCR(
                det_box_type='quad',
                det_model_dir=os.path.join(
                    os.path.dirname(__file__),
                    r'./inference/Student2',
                ),
                **extra_args,
            )
        rtn_map = {}
        for task_index,task_key in enumerate(task_chain):
            engine:PaddleOCR = task_chain[task_key]
            result_set = engine.ocr(img_array, cls=True,rec=False)
            formatted = []
            def recovery_points(x, y):
                new_x = x/scale
                new_y = y/scale
                return [round(new_x),round(new_y)]
            for result in result_set:
                for line in result:
                    text_val = ''
                    text_confidence = 0.0
                    restored_box = [recovery_points(*item) for item in copy.copy(line)]
                    formatted.append({
                        'text': text_val,
                        'confidence': text_confidence,
                        'points': restored_box,
                    })
            rtn_map[task_key] = formatted
        if not len(rtn_map):
            return jsonify(data=None)
        return jsonify(data=rtn_map,cn_msg='',en_msg='')
    except Image.DecompressionBombError:
        traceback.print_exc()
        return jsonify(error="Image size exceeds security limit")
    except Exception:
        traceback.print_exc()
        return flask.abort(404)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
