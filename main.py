from flask import Flask,render_template,url_for,request
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import pickle
import os
import numpy as np
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for


UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


app = Flask(__name__ , template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def home():
    return render_template('index.html')





@app.route('/second', methods=['GET','POST'])
def predict():
    
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
    training_set = train_datagen.flow_from_directory('Dataset/TRAINING',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory('Dataset/TEST',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=12, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    cnn.fit(x = training_set, validation_data = test_set, epochs = 1)
    
    '''
    model = pickle.load(open('recyclemodel.pkl', 'rb'))
    print(model)
    if request.method == 'POST':
        if model[0][0] == 1:
            prediction = 'r'
        else:
            prediction = 'nr'
            '''
    
    
    
    
    if request.method == 'POST':
        f = request.files['file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(path)

        # Save the file to ./uploads
        '''
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'single', secure_filename(f.filename))
        f.save(file_path)
        '''
        
        

        test_image = image.load_img(f, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = cnn.predict(test_image)
        training_set.class_indices
        if result[0][0] == 1:
            prediction = 'r'
        else:
            prediction = 'nr'
        
    return render_template('second.html', result = prediction)















if __name__ =='__main__':
    app.run(debug=True)