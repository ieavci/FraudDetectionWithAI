from flask import Flask
import os
app = Flask(__name__)


@app.template_filter('zip')
def zip_filter(*args):
    return zip(*args)

# Zip fonksiyonunu global olarak kaydetme
app.jinja_env.globals['zip'] = zip

# Import routes and other logic
from routes import *
from visualization import *
from model_training import *
from model_storage import *

# Model ayarlarının saklanacağı dizin
MODELS_DIR = "models/saved_models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

if __name__ == '__main__':
    app.run(debug=True)


