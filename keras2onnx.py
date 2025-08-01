import tensorflow as tf
import tf2onnx
from tensorflow import keras
from tensorflow.keras.models import load_model

model = load_model('NNeval.keras')
model.output_names=['output']


spec = (tf.TensorSpec((None, 832), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    output_path="NNeval.onnx"
)
