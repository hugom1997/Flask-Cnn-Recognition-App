import flask
from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
import base64
import numpy as np
import cv2
import tensorflow as tf

#Initialize the useless part of the base64 encoded image.
init_Base64 = 21;

#Our dictionary
label_dict = {0:'Cat', 1:'Giraffe', 2:'Sheep', 3:'Bat', 4:'Octopus', 5:'Camel'}


#Initializing the Default Graph (prevent errors)
graph = tf.get_default_graph()

# Use pickle to load in the pre-trained model.
with open(f'D:\Documents\Polytechnique\Trimestre 3\DrawApp\Drawing_App\model_cnn.pkl', 'rb') as f:
        model = pickle.load(f)

#Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__, template_folder='templates')

#First route : Render the initial drawing template
@app.route('/')
def home():
	return render_template('draw.html')



#Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():
        global graph
        with graph.as_default():
            if request.method == 'POST':
                    final_pred = None
                    #Preprocess the image : set the image to 28x28 shape
                    #Access the image
                    draw = request.form['url']
                    #Removing the useless part of the url.
                    draw = draw[init_Base64:]
                    #Decofing
                    draw_decoded = base64.b64decode(draw)
                    image = np.asarray(bytearray(draw_decoded), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
                    #Resizing and reshaping to keep the ratio.
                    resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
                    vect = np.asarray(resized, dtype="uint8")
                    vect = vect.reshape(1, 1, 28, 28).astype('float32')
                    #Launch prediction
                    my_prediction = model.predict(vect)
                    print(my_prediction)
                    #Getting the index of the maximum prediction
                    index = np.argmax(my_prediction[0])
                    print(index)
                    #Associating the index and its value within the dictionnary
                    final_pred = label_dict[index]

        return render_template('results.html', prediction =final_pred)


if __name__ == '__main__':
	app.run(debug=True)
