import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import boto3
import shutil

UPLOAD_FOLDER = '/home/ubuntu'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            s3 = boto3.resource('s3')
            bucket='flaskdataml2'
            data = open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb')
            response=s3.Bucket(bucket).put_object(Key=filename,Body=data)
            return redirect(url_for('upload_file',filename=filename))
    return  '''<!doctype html>
    <title>Face System</title>
    <h1>Upload a picture and see if it's a picture of database!</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''


@app.route('/output')
def show_prediction():
    download_data()
    download_path='/home/ubuntu/flask_app/static'
    files = [f for f in os.listdir(download_path) if os.path.isfile(os.path.join(download_path, f))]
    return render_template(
                'results.html', results=files)


def download_data():
    bucket='flaskdataml2'
    s3_client=boto3.client('s3')
    s3=boto3.resource('s3')
    download_path = '/home/ubuntu/flask_app/static'
    if(os.path.isdir(download_path)):
        shutil.rmtree(download_path)
    os.mkdir(download_path)
    download_path =download_path
    list1=s3_client.list_objects(Bucket=bucket)['Contents']
    print(list1)
    for key in list1:
        s3.Bucket(bucket).download_file(key['Key'], download_path+"/"+key['Key'])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
