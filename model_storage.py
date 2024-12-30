import os
import pickle

MODELS_DIR = "models/saved_models"

def save_model(model_id, model_info):
    file_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(model_info, file)

def load_model(model_id):
    file_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def list_saved_models():
    models = []
    for file_name in os.listdir(MODELS_DIR):
        if file_name.endswith('.pkl'):
            model_id = file_name.replace('.pkl', '')
            models.append(model_id)
    return models
