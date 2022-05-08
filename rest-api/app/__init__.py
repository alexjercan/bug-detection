from flask import Flask
from app.inference import Session

app = Flask(__name__)
ses = Session()
from app import views
