from flask import Blueprint, render_template, request
from app import model, kmeans_model, crop_cluster_map
from app.core.prediction import make_grouped_predictions

bp = Blueprint('routes', __name__)

@bp.route('/')
def home():
    return render_template('home.html')

@bp.route('/about')
def about():
    return render_template('about.html')

@bp.route('/findyourcrop')
def findyourcrop():
    return render_template('findyourcrop.html')

@bp.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template(
            'findyourcrop.html',
            prediction_text="Model not found! Please run train_model.py first."
        )

    try:
        N = int(request.form['N'])
        P = int(request.form['P'])
        K = int(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        result = make_grouped_predictions(
            model, kmeans_model, crop_cluster_map,
            N, P, K, temperature, humidity, ph, rainfall,
            top_n=3
        )

        return render_template(
            'findyourcrop.html',
            top_predictions=result["top_predictions"],
            group_crops=result["group_crops"],
            family_crops=result["family_crops"],
            crop_family=result["crop_family"]
        )

    except Exception as e:
        return render_template(
            'findyourcrop.html',
            prediction_text=f"Error: {str(e)}"
        )
