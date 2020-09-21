from flask import Flask
from flask_bootstrap import Bootstrap
from config import Config
from flask_wtf.csrf import CsrfProtect

csrf = CsrfProtect()


app = Flask(__name__)
app.config.from_object(Config)

bootstrap = Bootstrap(app)
csrf.init_app(app)

from app import routes
