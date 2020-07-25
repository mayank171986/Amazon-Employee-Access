from flask import render_template, jsonify, Flask, redirect, url_for, request
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import random
import numpy as np
from flask import Flask
import os
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
import glob


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = 'static/upload_folder/'
target_size = (256, 256, 3)

def get_predictions(path):
    img = image.load_img(path , target_size=(256,256))
    img = image.img_to_array(img)
    img = img.reshape((1,) + img.shape)
    img = img / 255.0
    img = img.reshape(1,256,256,3)
    classes = model.predict(img)
    return classes[0][0]

def inception():
    input_tensor = Input(shape=target_size)
    base_model = InceptionV3(include_top=False, weights= None, input_shape=target_size)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model

model = inception()
model.load_weights('weights.hdf5')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')
  
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    try:
        files = glob.glob('static/upload_folder/')
        for f in files:
            os.remove(f)
    except:
        pass

    if request.method == 'POST':
        if request.files:
            file = request.files['fileToUpload']
            filename = secure_filename(file.filename)


            path = os.path.join('upload_folder/', filename)

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            output = get_predictions('static/upload_folder/'+filename)
            output = output*100
            normal_out = '{:.2f}%'.format(100-output)
            pneumonia_out  = '{:.2f}%'.format(output)


            dic = {"PNEUMONIA":pneumonia_out, "NORMAL":normal_out}
            return render_template('uploaded.html', results = dic, path = path)

if __name__ == '__main__':
	# Threaded option to enable multiple instances for multiple user access support
	app.run(debug = True,threaded=True, port=8000)
