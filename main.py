import os
import cv2
import hashlib
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, send_file, url_for, render_template, jsonify

from config import *
from censorLicensePalate import *
from modelYOLO import *
from modelFasterRCNN import *

# Load model
app = Flask(__name__)    


# Check type HTTP
if CFG_HTTP_TYPE == 'ngrok':
    from flask_ngrok import run_with_ngrok
    run_with_ngrok(app)


# Config App
app.config['UPLOAD_FOLDER'] = CFG_PATH_UPLOAD
app.config['MAX_CONTENT_LENGTH'] = CFG_MAX_CONTENT_LENGTH

# Hash file name
def convertFileName(fileName):
    return hashlib.md5((str(datetime.now().time()) + "_" + fileName).encode()).hexdigest() + ".jpg"

# Check allow file upload
def allowedFile(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in CFG_ALLOWED_EXTENSIONS
    
# Upload file
def uploadFile(requestFile, pathSave, preFix=""):
    if requestFile.filename == '':
        return None, None
    if requestFile and allowedFile(requestFile.filename):
        fileName = preFix + convertFileName(secure_filename(requestFile.filename))
        pathSaveFile = os.path.join(pathSave, fileName)
        requestFile.save(pathSaveFile)
        return fileName, pathSaveFile
    return None, None
	
@app.route('/')
def default():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def home():
    model = None
    CFG_MODEL = request.form['model']  # Get the model selected by the user
    if CFG_MODEL == 'yolo':
        model = modelYOLO()
    else:
        model = modelFasterRCNN()
    if 'image' not in request.files:
        return redirect(request.url)
    fileName, pathImage = uploadFile(request.files['image'], CFG_PATH_UPLOAD)
    if pathImage is None:
        return redirect(request.url)
    image = cv2.imread(pathImage)
    listCoors = model.predict(image)
    
    imageResult = None
    typeBlur = int(request.form['typeBlur'])
    if typeBlur == 3: # Replace Image
        if 'imageReplace' not in request.files:
            return redirect(request.url)
        fileNameReplace, pathImageReplace = uploadFile(request.files['imageReplace'], CFG_PATH_UPLOAD)
        if pathImageReplace is None:
            return redirect(request.url)
        imageReplace = cv2.imread(pathImageReplace)
        imageResult = convertImage(image, listCoors).replaceImage(imageReplace)
    else:
        kernelSize = int(request.form['kernelSize'])
        if typeBlur == 1: # gaussianBlur
            imageResult = convertImage(image, listCoors).gaussianBlur(kernelSize)
        elif typeBlur == 2: # medianBlur
            imageResult = convertImage(image, listCoors).medianBlur(kernelSize)
    pathImageResult = os.path.join(CFG_PATH_RESULT, fileName)
    cv2.imwrite(pathImageResult, imageResult)
    return render_template('home.html', filename=fileName)

# Rule Show Image Src
@app.route('/showSrc/<filename>')
def displayImageSrc(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
# Rule Show Image Des 
@app.route('/showDes/<filename>')
def displayImageDes(filename):
	return redirect(url_for('static', filename='results/' + filename), code=301)

# Rule Download Image
@app.route('/download/<filename>')
def downloadFile(filename):
    return send_file(os.path.join(CFG_PATH_RESULT, filename), as_attachment=True)

# Main
if __name__ == "__main__":
    app.debug = True
    app.run()
