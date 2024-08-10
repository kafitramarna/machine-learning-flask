from flask import Flask
from Blueprints.regression import regression_bp

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'

app.register_blueprint(regression_bp, url_prefix='/regression')

