from board import Board, load_dictionary
from preprocessing import get_all_letters_in_image, read_im
import numpy as np
from keras.models import load_model

import os
from flask import render_template, flash, redirect, url_for, session, request
from app import app
import uuid
from werkzeug.utils import secure_filename
from app.forms import PhotoForm, BoardForm, LetterForm, RowLetterForm
import tensorflow as tf

D = None
model = None
key = None

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


@app.route('/', methods=['GET','POST'])
@app.route('/index', methods=['GET','POST'])
def index():
    form = PhotoForm()
    session['letters'] = None
    if form.validate_on_submit():
        f = form.photo.data
        filename = secure_filename(f.filename)
        filename = os.path.join(app.instance_path, 'photos', filename)
        os.makedirs(os.path.dirname(filename),exist_ok=True)
        f.save(filename)
        session['current_img'] = filename
        return redirect(url_for('result'))
    return render_template('index.html', form=form)

@app.route('/result', methods=['GET', 'POST'])
def result():
    global D,key,model
    if D is None:
        D = load_dictionary(app.config['DICTIONARY_FN'])


    if key is None:
        alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L',\
          'M','N','O','P','QU','R','S','T','U','V','W','X','Y','Z']
        key = dict(zip(range(0,len(alphabet)),alphabet))

    if model is None:
        model = load_model(app.config["MODEL_FN"])

    fn = session['current_img']

    letters = session.get('letters')
    if letters is None:
        print("getting letters")
        letters,labels,image_data = get_all_letters_in_image(fn,model,key,\
                                        app.config["IMG_X"],app.config["IMG_Y"],\
                                        (app.config["BOARD_X"],app.config["BOARD_Y"]))
    else:
        print('not getting letters')
        _,_,image_data = get_all_letters_in_image(fn,model,key,\
                                        app.config["IMG_X"],app.config["IMG_Y"],\
                                        (app.config["BOARD_X"],app.config["BOARD_Y"]))

    # letters = [('E', 'H', 'QU', 'F'), ('S', 'A', 'F', 'E'), ('C', 'A', 'U', 'R'),\
    #             ('L', 'A', 'M', 'E')]

    if letters is None:
        flash('Take another picture!!')
        return redirect(url_for('index'))


    b = Board(app.config["BOARD_X"],app.config["BOARD_Y"],letters,D)
    b.search()
    words = b.get_words()

    form = BoardForm()

    for row_id,row in zip([form.row1,form.row2, form.row3, form.row4],letters):
        for value in row:
            letterform = LetterForm()
            letterform.letter = value
            row_id.append_entry(letterform)

        # for r in rowform.row:
        #     print(r.data)
    if form.validate_on_submit():
        new_letters = []
        for row in [form.row1,form.row2, form.row3, form.row4]:
            r = []
            for val in row[:4]:
                r.append(val.letter.data)
            new_letters.append(tuple(r))

        if letters == new_letters:
            save_data(image_data,np.asarray(new_letters).reshape(-1))
            session['letters'] = None
            session['image_data'] = None
            return redirect(url_for('index'))
        else:
            print("refresh words")
            session['letters'] = new_letters
            return redirect(url_for('result'))

    return render_template('result.html',words=words,form=form)

def save_data(new_x,new_y):

    if os.path.exists(app.config["LABELS_FN"]):
        labels = np.load(app.config["LABELS_FN"])
        images = np.load(app.config["IMAGES_FN"])

        labels = labels['labels']
        images = images['images']

        np.append(labels,new_y,axis=0)
        np.append(images,new_x,axis=0)
    else:
        labels = new_y
        images = new_x

    np.savez(app.config["LABELS_FN"],labels=labels)
    np.savez(app.config["IMAGES_FN"],images=images)
