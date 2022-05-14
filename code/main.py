import os
from flask import Flask, render_template, request,redirect
from werkzeug.utils import secure_filename
import albumentations as A
from PIL import Image
import numpy as np
import cv2 as cv
import h5py
from keras.models import load_model

#################

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"

#################
#helper functions
def augmentImage(img):
    augmentationResult= A.Compose([
        A.to_float(img=img,max_value=255)
    ])
    print(augmentationResult)
    return augmentationResult

def resizeImage(img):
    image =cv.resize(img,(300,300),3)
    return image

def crop_center(img,cropX,cropY):
    y,x,_ = img.shape
    startX = x//2-(cropX//2)
    startY = y//2-(cropY//2)    
    return img[startY:startY+cropY,startX:startX+cropX]


def transformInput(input):
    # 3. albumentations
    # 2. resize
    # 1. crop center
    # 0. reshape
    output=np.reshape(crop_center(resizeImage(input),224,224),(1,224,224,3)) / 255
    return output


def Predict(filename):
    result=""
    
    inputImg=Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # inputImg.show()
    
    # converting image to array
    arrImg= np.asfarray(inputImg,dtype=np.float32) / 255
    # 1. transfrom image
    transformedImage=transformInput(arrImg)
    # 2. load model 
    model = load_model('wrist_model_v1.h5')
    # 3. predict
    pred=np.argmax(model.predict(transformedImage))
    print("################################")
    print(pred)
   
    # 4. calculation of what to show
    if pred == 0: result="Bone is Fractured"
    else: result="Bone is not Fractured"
    
    # 5. send output back
    return result
    
#################


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html",status=0,message='')

@app.route('/handle-upload', methods=['POST'])
def handle_upload():
    try:
        file=request.files['file']
        # check if the post request has the file part
        if 'file' not in request.files or file.filename == '':
            return render_template("index.html",message="Please select an xray image to upload",msg_type="error")
        
        if request.method=="POST" and "file" in request.files:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
            # first saving image then opening it
            result=Predict(filename) 
        
        return render_template("index.html",message=result,msg_type="success",status=1)
    
    except:
        return render_template("index.html",message="Some Internal Error Occurred !! Please Try Again",msg_type="error",status=1)


if __name__ == '__main__':
    app.run(debug=True)
