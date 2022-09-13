import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, request, send_from_directory, send_file, jsonify
from werkzeug.utils import secure_filename

import cv2
import numpy as np
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import easyocr

from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from predict import extract_features, generate_desc

IMG_SIZE = (250, 500)
NUM_CLASSES = 7
BATCH_SIZE = 6
NUM_EPOCH = 15
FREEZE_LAYERS = 16
LEARNING_RATE = 0.0002
DROP_OUT = .2
class_dictionary = {'10': 0, '100': 1, '20': 2, '200': 3, '2000': 4, '50': 5, '500': 6}
vals = list(class_dictionary.values())
keys = list(class_dictionary.keys())

model = Xception(include_top = False,
              weights = 'imagenet',
              input_tensor = None,
              input_shape = (250, 500, 3))

top_layer = model.output
x = GlobalAveragePooling2D()(top_layer)
op = Dense(NUM_CLASSES, activation = 'softmax', name = 'softmax')(x)

model_final = Model(inputs = model.input, outputs = op)
for layer in model_final.layers[:FREEZE_LAYERS]:
  layer.trainable = False

for layer in model_final.layers[FREEZE_LAYERS:]:
  layer.trainable = True

model_final.compile(optimizer = Adam(lr = LEARNING_RATE),
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])

model_final.load_weights('static/Xception_model.h5')
#reader = easyocr.Reader(['en'])

#COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1
app.config['IMAGE_FOLDER'] = os.path.abspath('.')+'/static'
ALLOWED_EXTENSIONS=set(['png','jpg','jpeg','gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

'''
@app.route('/')
def man():
    return render_template('index.html')
'''

@app.route('/home',methods=['POST'])
def upload_file():
    if request.method=='POST':
        digit = ""
        for k in request.files:
            file = request.files[k]
            if file and allowed_file(file.filename):
                filename=secure_filename(file.filename)
                file.save(os.path.join(app.config['IMAGE_FOLDER'],filename))
            filename=secure_filename(file.filename)
            img_arr = cv2.imread('static/{}'.format(filename))
            img_arr = cv2.resize(img_arr, (500,250))
            test_image = np.expand_dims(img_arr, axis=0)
            test_image = preprocess_input(test_image)
            prediction = model_final.predict(test_image)
            idx = np.argmax(prediction, axis=1)
            digit = keys[vals.index(idx)]
        
        return jsonify({"code":1,"prediction":digit})
    
'''
@app.route('/docread',methods=['POST'])
def doc_reader():
    if request.method=='POST':
        for k in request.files:
            file = request.files[k]
            if file and allowed_file(file.filename):
                filename=secure_filename(file.filename)
                file.save(os.path.join(app.config['IMAGE_FOLDER'],filename))
            filename=secure_filename(file.filename)
            output = reader.readtext('static/{}'.format(filename))
            paragraph = ''
            i = 0
            while i < len(output):
                paragraph += output[i][1] + " "
                i += 1
        
        return jsonify({"code":1,"prediction":paragraph[0,5]})
'''

tokenizer = load(open('D:\Aries\Image Captioning/tokenizer.pkl', 'rb'))
max_length = 34
model1 = load_model('D:\Aries\Image Captioning/model-ep003-loss3.647-val_loss3.871.h5')    


@app.route('/caption',methods=['POST'])
def caption_generator():
    if request.method=='POST':
        description = ""
        for k in request.files:
            file = request.files[k]
            if file and allowed_file(file.filename):
                filename=secure_filename(file.filename)
                file.save(os.path.join(app.config['IMAGE_FOLDER'],filename))
            filename=secure_filename(file.filename)
            photo = extract_features('static/{}'.format(filename))
            description = generate_desc(model1, tokenizer, photo, max_length)
            description = description[9:len(description)-7]
            
        return jsonify({"code":1,"prediction":description})




@app.route("/static/<imgname>",methods=['GET'])
def images(imgname):
    return send_from_directory(app.config['IMAGE_FOLDER'],imgname)


if __name__ == '__main__':
    if not os.path.exists(app.config['IMAGE_FOLDER']):
        os.mkdir(app.config['IMAGE_FOLDER'])
    app.run(debug=True)