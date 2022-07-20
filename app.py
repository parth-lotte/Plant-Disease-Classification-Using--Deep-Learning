from cv2 import COLOR_BGR2GRAY
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from keras_preprocessing import image 
from flask import request, jsonify
from flask import Flask, render_template, url_for


app = Flask(__name__)
model = Model
model = load_model('VGG.h5')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
    
@app.route('/',methods=['POST'])
def predict():
    imgfile = request.files['imgfile']
    image_path = "./static/uploads/" + imgfile.filename
    imgfile.save(image_path)

    # img = image.load_img(image_path, target_size=(224, 224))
    # img = image.img_to_array(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = np.expand_dims(img, axis=0)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = image.img_to_array(img)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)


    prediction = model.predict(img)
    result = np.argmax(prediction,axis=1)

    if result == [0]:
        ans = "Healthy"
    elif result == [1]:
        ans = "Powdery"
    elif result == [2]:
        ans = "Rust"
    else :
        ans = "Error"




    return render_template('result.html', prediction_text=ans, imgurl=image_path)
    # return render_template('result.html')

# @app.route('/result')
# def result():
#     return render_template('result.html')


if __name__ == "__main__":
    app.run(debug=True)