import os
import joblib
from flask import Flask
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Determine project base directory (parent of app package)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define custom paths to template and static folders to match the target repository layout
template_dir = os.path.join(BASE_DIR, 'templates')
static_dir = os.path.join(BASE_DIR, 'static')

# Create Flask application instance
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Load machine learning model
MODEL_PATH = os.path.join(BASE_DIR, 'crop_model.pkl')
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully from:", MODEL_PATH)
else:
    model = None
    print("Model file not found at:", MODEL_PATH)

# Load K-Means clustering model and crop -> group map
# (used to recommend other crops that share similar soil/environmental
# conditions as the predicted crop)
KMEANS_PATH = os.path.join(BASE_DIR, 'kmeans_model.pkl')
CROP_CLUSTER_MAP_PATH = os.path.join(BASE_DIR, 'crop_cluster_map.pkl')

if os.path.exists(KMEANS_PATH) and os.path.exists(CROP_CLUSTER_MAP_PATH):
    kmeans_model = joblib.load(KMEANS_PATH)
    crop_cluster_map = joblib.load(CROP_CLUSTER_MAP_PATH)
    print("Clustering artifacts loaded successfully.")
else:
    kmeans_model = None
    crop_cluster_map = None
    print("Clustering artifacts not found. Run train_model.py to generate them.")

# Register routes (imported here to avoid circular dependencies)
from app.api import routes
app.register_blueprint(routes.bp)
