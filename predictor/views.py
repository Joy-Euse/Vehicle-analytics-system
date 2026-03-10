from django.shortcuts import render
from model_generators.clustering.train_cluster_optimized import evaluate_clustering_model
from model_generators.classification.train_classifier import evaluate_classification_model
from model_generators.regression.train_regression import evaluate_regression_model
from predictor.data_exploration import dataset_exploration, data_exploration, rwanda_map_exploration
import pandas as pd
import joblib
import numpy as np

regression_model = joblib.load("model_generators/regression/regression_model.pkl")
classification_model = joblib.load("model_generators/classification/classification_model.pkl")
clustering_kmeans = joblib.load("model_generators/clustering/clustering_kmeans.pkl")  # KMeans for predictions
clustering_scaler = joblib.load("model_generators/clustering/clustering_scaler.pkl")
clustering_features = joblib.load("model_generators/clustering/clustering_features.pkl")

def data_exploration_view(request):
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    context = {
        "data_exploration": data_exploration(df),
        "dataset_exploration": dataset_exploration(df),
        "rwanda_map": rwanda_map_exploration(df),
        }
    
    return render(request, "predictor/index.html", context)

def regression_analysis(request):
    context = {
        "evaluations": evaluate_regression_model()
        }
    
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        prediction = regression_model.predict([[year, km, seats, income]])[0]
        context["price"] = prediction
    
    return render(request, "predictor/regression_analysis.html", context)




def classification_analysis(request):
    context = {
        "evaluations": evaluate_classification_model()
    }
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        prediction = classification_model.predict([[year, km, seats, income]])[0]
        context["prediction"] = prediction
        return render(request, "predictor/classification_analysis.html", context)
    
    return render(request, "predictor/classification_analysis.html", context)


def clustering_analysis(request):
    context = {
        "evaluations": evaluate_clustering_model()
    }

    if request.method == "POST":
        try:
            # Get input values (regression features)
            year = int(request.POST["year"])
            km = float(request.POST["km"])
            seats = int(request.POST["seats"])
            income = float(request.POST["income"])

            # Step 1: Predict price using regression model
            predicted_price = regression_model.predict([[year, km, seats, income]])[0]
            
            # Step 2: Prepare extended features for clustering
            # Features: ["estimated_income", "selling_price", "year", "kilometers_driven", "seating_capacity"]
            clustering_input = np.array([[
                income,                    # estimated_income
                predicted_price,           # selling_price
                year,                      # year
                km,                        # kilometers_driven
                seats                      # seating_capacity
            ]])
            
            # Step 3: Scale the features
            features_scaled = clustering_scaler.transform(clustering_input)
            
            # Step 4: Predict cluster using KMeans
            cluster_id = clustering_kmeans.predict(features_scaled)[0]
            
            # Mapping for KMeans(k=2) clusters
            cluster_mapping = {
                0: "Economy",
                1: "Standard"
            }
            
            cluster_name = cluster_mapping.get(cluster_id, f"Segment_{cluster_id}")

            context.update({
                "prediction": cluster_name,
                "price": predicted_price
            })
        except Exception as e:
            context["error"] = str(e)
    return render(request, "predictor/clustering_analysis.html", context)