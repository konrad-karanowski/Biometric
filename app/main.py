from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
import cv2
import time

from app.compare_face import compare_face

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = '/home/mzieba/Biometrics/arcface-pytorch/app/database'
DATABASE = '/home/mzieba/Biometrics/arcface-pytorch/app/users.csv'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


user_df = pd.DataFrame(columns=['name', 'photo'])


# Check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def index():
    return render_template('index.html')

from werkzeug.utils import secure_filename

import numpy as np

# Function to detect faces in an image and return their coordinates
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

# Function to crop the image around the detected face
def crop_face(image, face_coordinates):
    x, y, w, h = face_coordinates
    return image[y:y+h, x:x+w]


@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    global user_df
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        name = request.form['name']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load the image
            image = cv2.imread(filepath)

            os.remove(filepath)
            
            # Detect faces in the image
            faces = detect_faces(image)
            
            if len(faces) > 0:
                # Crop the image around the first detected face
                cropped_image = crop_face(image, faces[0])
                
                # Save the cropped image
                cropped_filename = 'cropped_' + filename
                cropped_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cropped_filename)
                cv2.imwrite(cropped_filepath, cropped_image)
                
                # Append user information to the DataFrame
                try:
                    user_df = pd.read_csv(DATABASE)
                    user_df = pd.concat([user_df, pd.DataFrame({'name': [name], 'photo': [cropped_filename]})], ignore_index=True)
                except:
                    user_df = pd.DataFrame({'name': [name], 'photo': [cropped_filename]})
                user_df.to_csv(DATABASE, index=False)
                
                flash('User added successfully')
                return redirect(url_for('index'))
            else:
                flash('No face detected in the uploaded photo')
                return redirect(request.url)
    
    return render_template('add_user.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        name = request.form['name']

        df = pd.read_csv(DATABASE)
        try:
            user_photo = cv2.imread(os.path.join(UPLOAD_FOLDER, df[df.name == name].photo.iloc[0]))
            user_photo = user_photo[... ,::-1]
        except Exception as e:
            print(e)
            flash('No such user')
            return redirect(url_for('index'))
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load the image
            image = cv2.imread(filepath)
            

            st = time.time()

            # Detect faces in the image
            faces = detect_faces(image)
            
            if len(faces) > 0:
                # Crop the image around the first detected face

                ep1 = time.time()
                cropped_image = crop_face(image, faces[0])
                cropped_image = cropped_image[... ,::-1]
                
                comparison, sim = compare_face(cropped_image, user_photo)
                print(sim)

                et = time.time()
                time_info = f'Total time: {round(et - st, 4)}. Detection: {round(ep1 - st, 4)}. Classification: {round(et - ep1, 4)}'

                if comparison:
                    flash(f'Welcome, {name}! Similarity: {round(sim, 2)}. {time_info}')
                else:
                    flash(f'Login failed. Face not recognized. Similarity: {round(sim, 2)}. {time_info}')
                
                return redirect(url_for('index'))
            else:
                flash('No face detected in the uploaded photo')
                return redirect(url_for('index'))
    
    return render_template('login.html')


if __name__ == '__main__':
    app.run(debug=True)
