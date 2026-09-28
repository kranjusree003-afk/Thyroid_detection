import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)

model = tf.keras.models.load_model("thyroid_cancer_detection.keras",compile=False)

class_names = ['2', '3', '4A', '4B', '4C', '5']

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            img = Image.open(file).convert("RGB")
            img = img.resize((128, 128))

            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            result = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(result[0])

            prediction = class_names[predicted_class]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
