from flask_wtf import FlaskForm
from wtforms import SubmitField,StringField,FieldList, FormField
from wtforms.validators import DataRequired,Length

from flask_wtf.file import FileField, FileRequired

class LetterForm(FlaskForm):
    letter = StringField()

class RowLetterForm(FlaskForm):
    row = FieldList(FormField(LetterForm),min_entries=1)

class BoardForm(FlaskForm):
    row1 = FieldList(FormField(LetterForm),min_entries=1)
    row2 = FieldList(FormField(LetterForm),min_entries=1)
    row3 = FieldList(FormField(LetterForm),min_entries=1)
    row4 = FieldList(FormField(LetterForm),min_entries=1)
    # rows = FieldList(FormField(RowLetterForm),min_entries=1)
    submit = SubmitField('Submit')

class PhotoForm(FlaskForm):
    photo = FileField(validators=[FileRequired()])
    submit = SubmitField('Submit')
