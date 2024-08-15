from flask import Flask
from regression import regression_bp
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Mengaktifkan CORS untuk seluruh aplikasi
@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'

app.register_blueprint(regression_bp, url_prefix='/regression')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)