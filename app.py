import numpy as np
from flask import Flask, request,render_template,jsonify
import pickle
from flask_mail import Mail
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import cv2 as cv
import tensorflow as tf
import keras.backend as k
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from PIL import Image
import pytesseract as py
from flask_mail import Message

app = Flask(__name__)
mail=Mail(app)
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = '********@gmail.com'
app.config['MAIL_PASSWORD'] = '***********'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)





model = pickle.load(open('model.pkl', 'rb'))
model_loan = pickle.load(open('loan_model.pkl', 'rb'))
model_text = pickle.load(open('text_model.pkl', 'rb'))
model_hr = pickle.load(open('hr_model.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/image_classification')
def image():
    return render_template('image_classification.html')

@app.route('/loan')
def loan():
    return render_template('loan.html')

@app.route('/loan',methods=['POST'])
def loan_predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model_loan.predict(final_features)

    output = prediction[0]
    if int(output)==1:
        prediction1="Yes!!! You Are Eligible For This Loan"
    else:
        prediction1="No You Are Not Eligible For This Loan"
    return render_template('loan.html',
   prediction_text=prediction1)

@app.route('/image_classification',methods=['POST'])
def image_classification():
    file = request.files['image']
    extension = os.path.splitext(file.filename)[1]
    oldname = os.path.splitext(file.filename)[0]
    f_name=oldname+extension
    app.config['UPLOAD_FOLDER'] = 'static/upload'
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name))
    
    ipath='static/upload'
    img_path=os.path.join(ipath, f_name)
    
    
    
    
    
    img=cv.imread(img_path)
    #print(img)
    #img = cv.resize(img, (150, 150))
    #print(img)
    #img=img/255.
    img = np.expand_dims(img.reshape(-1,150,150),0)
    #print(img)
    pred=model.predict(img)
    #_pred=load_img(img_path,target_size=(150,150))
   #img_pred=img_to_array(img_pred)
    # Reshape the image into a sample of 1 channel
    #img_pred = img_pred.reshape(1,150, 150,3)
    # Prepare it as pixel data
    #img_pred = img_pred.astype('float32')
    #img_pred = img_pred / 255.0
    
   #img_pred=np.expand_dims(img_pred,axis=0)
   #img_pred=preprocess_input(img_pred,mode='caffe')
    #class_labels = {0: 'Cat', 1: 'Dog',2:'Elephant'}
    #img_prediction = model_predict(img_path,model)
    #guess = prediction[0]
    #result=model.predict_classes(img_pred)
    #pred_class = decode_predictions(img_prediction, top=1)   # ImageNet Decode
    #result = str(pred_class[0][0][1]) 
    
    return render_template('image_classification.html',
                           prediction_image=pred)

@app.route('/text')
def text():
    return render_template('text.html')

@app.route('/text_predict',methods=['POST'])
def text_predict():

    name =request.form["name"]
    #final_features = np.array(int_features)
    prediction = model_text.predict([name])
    result=str(prediction)
 
    if name:
        msg = result
        return jsonify({'name' : msg})
 
    return jsonify({'error' : 'Missing data!'})

@app.route('/hr')
def hr():
    return render_template('hr.html')

@app.route('/hr',methods=['POST'])
def hr_predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model_hr.predict(final_features)

    output = prediction[0]
    if int(output)==1:
        prediction1="Yes!!! You Are Eligible For Promotion"
    else:
        prediction1="No You Are Not Eligible For Promotion"
    return render_template('hr.html',
   prediction_text=prediction1)

@app.route('/tesseract')
def image_text():
    return render_template('tesseract.html')
@app.route('/tesseract',methods=['POST'])
def text_image():
    file = request.files['image']
    extension = os.path.splitext(file.filename)[1]
    oldname = os.path.splitext(file.filename)[0]
    f_name=oldname+extension
    app.config['UPLOAD_FOLDER'] = 'static/upload'
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name))
    
    ipath='static/upload'
    img_path=os.path.join(ipath, f_name)
    tesseract_image=Image.open(img_path)
    text=py.image_to_string(tesseract_image)
    
    return render_template('tesseract.html',
                           prediction_image=text)
    
@app.route("/sent_mail",methods=['POST'])
def sent_mail():
   nam = request.form["n"]
   email = request.form["e"]
   subject = request.form["s"]
   feedback = request.form["f"]
   star = request.form["r"]
   msg = Message(subject, sender =email, 
                 recipients = ['*********@gmail.com'])
   msg.body = "Hello AV,\n"+feedback+"\nRating "+star+"\nFrom\n"+nam+"\nEmailId: "+email
   #mail.send(msg)
   if nam and email and subject and feedback and star:
       mail.send(msg)
       #n11=nam+email+subject+feedback+star
       return jsonify({"result":"👍 Thank You For Your Support!"})
       
   return jsonify({"error":"Incorrect Details"})






if __name__ == "__main__":
    app.run(debug=True)
