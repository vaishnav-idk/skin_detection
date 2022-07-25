from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import keras as kera
import numpy as np

app = Flask(__name__)

SKIN_CLASSES = {
    0: 'Acne',
    1: 'Malanoma',
    2: 'Psoriasis'

}

model = load_model('model.h5')


def predict_label(img_path):
    i = tf.keras.utils.load_img(img_path, target_size=(245, 245))
    i = tf.keras.utils.img_to_array(i) / 225.0
    i = i.reshape(1, 245, 245, 3)
    p = model.predict(i)
    print(type(p))
    return SKIN_CLASSES[np.argmax(p)]




# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")



@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)

    return render_template("index.html", prediction=p, img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)
