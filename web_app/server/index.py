from flask import Flask
from flask import make_response, jsonify
from flask_cors import CORS
from encoder import Encoder
from decoder import Decoder
from bahdanau_attention_layer import BahdanauAttention
from model import Model
import tensorflow as tf
import utils

app = Flask(__name__)
CORS(app)

@app.route('/translate/<en_sentence>/<model_type>/<beam_width>', methods=['GET'])
def translate(en_sentence, model_type, beam_width):
    beam_width = int(beam_width)
    if model_type == 'bahdanau':
        vi_sentence_gready = bahdanau_model.basic_translate(str(en_sentence))
        vi_sentence_beam, beam_score = bahdanau_model.beam_translate(str(en_sentence), beam_width=beam_width)
    else:
        vi_sentence_gready = luong_model.basic_translate(str(en_sentence))
        vi_sentence_beam, beam_score = luong_model.beam_translate(str(en_sentence), beam_width=beam_width)
    vi_sentence_gready = vi_sentence_gready[0].replace('_', ' ').replace(' </s>', '')
    response = make_response(
        jsonify(
            {"data": {
                'gready': vi_sentence_gready,
                'beam': vi_sentence_beam
            }}
        ),
        200,
    )
    response.headers["Content-Type"] = "application/json"
    return response

@app.route('/get-history',methods=['GET'])
def get_history():
    return 'history'

if __name__ == "__main__":
    global luong_model 
    global bahdanau_model 
    luong_model = Model()
    luong_model.load_model(checkpoint_dir='checkpoints/best_cp/luong/ckpt-7', dataset_file='checkpoints/best_cp/luong/infor_luong.pickle')
    
    bahdanau_model = Model()
    bahdanau_model.load_model(checkpoint_dir='checkpoints/best_cp/bahdanau/ckpt-5', dataset_file='checkpoints/best_cp/bahdanau/infor_bahdanau.pickle')
    app.run()