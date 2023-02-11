from app.inference import Session
from flask import Flask

app = Flask(__name__)
ses = Session()
from app import views
