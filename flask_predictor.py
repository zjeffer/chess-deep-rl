import base64
from typing import Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import config

# gpu memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

model = load_model(config.MODEL_FOLDER + "/model.h5")
app = Flask(__name__)



# CORS(app)

@tf.function
def predict(args) -> Tuple[list[float], float]:
	return model(args)

@app.route("/predict", methods=["POST"])
def make_prediction():
	data = request.json['data']
	data = np.frombuffer(base64.b64decode(data), dtype=bool)
	data = data.reshape(1, *config.INPUT_SHAPE)
	p, v = predict(data)
	p, v = p[0].numpy(), float(v[0][0])
	return jsonify({"prediction": p.tolist(), "value": v})

if __name__ == '__main__':
	app.run(host='localhost', port=5000)