from recognition import app, db
import os
from flask import render_template, redirect, url_for
from flask_login import current_user, login_user, login_required
from recognition.models import Recognition
from recognition.forms import RecognitionForm
from recognition.recognizer import test_model
from PIL import Image
import secrets
import numpy as np
from gtts import gTTS
from IPython.display import Audio



def save_recognition_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex
    picture_path = os.path.join(
        app.root_path, 'static\\pictures\\', picture_fn)
    output_size = (700,700)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path + ".jpg")
    return picture_fn

@app.route('/', methods=['GET','POST'])
@app.route('/recognize', methods=['GET','POST'])
def recognize():
    form = RecognitionForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_recognition_picture(form.picture.data)
            picture_prediction=test_model(os.path.dirname(__file__)+'\\static\\pictures\\'+picture_file + ".jpg")
            audio_path = os.path.join(app.root_path, 'static\\audio\\', picture_file + ".mp3")
            tts = gTTS(picture_prediction, lang='en', slow=True)
            tts.save(audio_path)
            recognition = Recognition(picture_name=picture_file, picture_prediction=picture_prediction)
            db.session.add(recognition)
            db.session.commit()
            login_user(recognition)
            return redirect(url_for('prediction')) 
        else:
            return redirect(url_for('recognize'))
    return render_template('recognize.html', form=form)


@app.route('/prediction',methods=['GET', 'POST'])
@login_required
def prediction():
    image_file = url_for(
        'static', filename='pictures/' + current_user.picture_name + ".jpg")
    audio_file = url_for(
        'static', filename='audio/' + current_user.picture_name + ".mp3")
    prediction = current_user.picture_prediction
    return render_template('prediction.html', image_file = image_file, prediction=prediction, audio_file = audio_file )

@app.route("/wav")
def streamwav():
    def generate():
        with open("signals/1.wav", "rb") as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
    return Response(generate(), mimetype="audio/x-wav")
