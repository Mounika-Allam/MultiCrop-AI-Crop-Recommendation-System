import numpy as np


CROP_FAMILIES = {
    "rice": "Cereals & High-Rainfall Crops",
    "maize": "Cereals & Warm-Climate Crops",
    "chickpea": "Legumes & Low-Rainfall Crops",
    "kidneybeans": "Legumes & Moderate-Rainfall Crops",
    "lentil": "Legumes & Low-Rainfall Crops",
    "mothbeans": "Legumes & Low-Rainfall Crops",
    "mungbean": "Legumes & Low-Rainfall Crops",
    "blackgram": "Legumes & Warm-Climate Crops",
    "pigeonpeas": "Legumes & Warm-Climate Crops",
    "banana": "Fruits & Humid-Climate Crops",
    "coconut": "Fruits & Tropical Crops",
    "coffee": "Commercial & Warm-Climate Crops",
    "cotton": "Fiber Crops & Dry-Climate Crops",
    "jute": "Fiber Crops & High-Rainfall Crops",
    "apple": "Fruits & Cool-Climate Crops",
    "grapes": "Fruits & Moderate-Climate Crops",
    "mango": "Fruits & Warm-Climate Crops",
    "orange": "Fruits & Moderate-Climate Crops",
    "papaya": "Fruits & Tropical Crops",
    "pomegranate": "Fruits & Low-Rainfall Crops",
    "muskmelon": "Fruits & Warm-Climate Crops",
    "watermelon": "Fruits & Warm-Climate Crops",
}

def make_prediction(model, n, p, k, temperature, humidity, ph, rainfall):
    """
    Executes model inference given soil and weather parameters.
    Returns the single best crop (kept for backward compatibility).
    """
    if model is None:
        raise ValueError("Model is not loaded. Please run train_model.py first.")
        
    features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)[0]
    return prediction.upper()


def make_top_predictions(model, n, p, k, temperature, humidity, ph, rainfall, top_n=3):
    """
    Executes model inference and returns the top_n most likely crops,
    ranked by predicted probability/confidence.

    Returns a list of dicts: [{"crop": "RICE", "confidence": 87.4}, ...]
    """
    if model is None:
        raise ValueError("Model is not loaded. Please run train_model.py first.")

    if not hasattr(model, "predict_proba"):
        # Fallback for models without probability support (e.g. plain LogisticRegression
        # without predict_proba enabled) - just return the single prediction.
        crop = make_prediction(model, n, p, k, temperature, humidity, ph, rainfall)
        return [{"crop": crop, "confidence": 100.0}]

    features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    # Sort indices by probability, descending
    top_indices = np.argsort(probabilities)[::-1][:top_n]

    results = []
    for idx in top_indices:
        results.append({
            "crop": classes[idx].upper(),
            "confidence": round(float(probabilities[idx]) * 100, 1)
        })
    return results


def make_grouped_predictions(model, kmeans_model, crop_cluster_map,
                              n, p, k, temperature, humidity, ph, rainfall,
                              top_n=3):
    """
    Returns the model's top predicted crop(s) PLUS every other crop that
    shares the same soil/environmental group (K-Means cluster) as the
    top predicted crop. This gives a longer, more useful list than just
    the top_n by raw probability - e.g. predicting "mothbeans" also
    surfaces the other legumes/dry-climate crops it clusters alongside.

    Returns a dict:
    {
        "top_predictions": [{"crop": "MOTHBEANS", "confidence": 91.0}, ...],
        "group_id": 0,
        "group_crops": ["BLACKGRAM", "CHICKPEA", ...]   # excludes the top crop itself
    }
    """
    top_predictions = make_top_predictions(
        model, n, p, k, temperature, humidity, ph, rainfall, top_n=top_n
    )

    top_crop = top_predictions[0]["crop"].lower()

    group_crops = []
    group_id = None

    if kmeans_model is not None and crop_cluster_map is not None:
        # Prefer grouping by the predicted crop's own known cluster
        # (based on how it behaved across the whole training set).
        if top_crop in crop_cluster_map:
            group_id = crop_cluster_map[top_crop]
        else:
            # Fallback: cluster the raw input features directly.
            features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
            group_id = int(kmeans_model.predict(features)[0])

        group_crops = sorted([
            crop.upper() for crop, cluster in crop_cluster_map.items()
            if cluster == group_id and crop != top_crop
        ])

    return {
        "top_predictions": top_predictions,
        "group_id": group_id,
        "group_crops": group_crops,
        "family_crops": [top_predictions[0]["crop"]] + group_crops,
        "crop_family": CROP_FAMILIES.get(
            top_crop, "Crops With Similar Soil & Climate Conditions"
        )
    }
