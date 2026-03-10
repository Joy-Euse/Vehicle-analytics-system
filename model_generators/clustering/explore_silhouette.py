import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
import joblib
import pandas as pd

df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")

print("=" * 80)
print("EXPLORING FEATURE COMBINATIONS FOR SILHOUETTE SCORE >= 0.9")
print("=" * 80)

# Test different feature combinations
feature_sets = {
    "Original (2 features)": ["estimated_income", "selling_price"],
    "Extended (5 features)": ["estimated_income", "selling_price", "year", "kilometers_driven", "seating_capacity"],
    "Income + Year + Seats": ["estimated_income", "year", "seating_capacity"],
    "Price + Year + KM": ["selling_price", "year", "kilometers_driven"],
    "All Numeric": ["estimated_income", "selling_price", "year", "kilometers_driven", "seating_capacity", "client_age"],
}

scalers = {
    "StandardScaler": StandardScaler(),
    "RobustScaler": RobustScaler(),
    "MinMaxScaler": MinMaxScaler(),
}

best_overall = {"score": -1, "config": None, "model": None, "scaler": None}

for feature_name, features in feature_sets.items():
    # Skip if features don't exist
    if not all(f in df.columns for f in features):
        print(f"\n⚠️  Skipping '{feature_name}' - missing columns: {[f for f in features if f not in df.columns]}")
        continue
    
    X = df[features].dropna()
    print(f"\n{'='*80}")
    print(f"Testing: {feature_name}")
    print(f"Features: {features}")
    print(f"Samples: {len(X)}")
    print(f"{'-'*80}")
    
    for scaler_name, scaler in scalers.items():
        X_scaled = scaler.fit_transform(X)
        print(f"\n  Testing {scaler_name}:")
        
        best_k_score = -1
        best_k_model = None
        best_k = None
        
        # Test KMeans with various k values
        for k in range(2, 12):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=5000)
                labels = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                
                if score > best_k_score:
                    best_k_score = score
                    best_k_model = kmeans
                    best_k = k
                
                if score >= 0.88:  # Print high scores
                    print(f"    KMeans(k={k}): {score:.4f} ⭐")
                    
                    if score > best_overall["score"]:
                        best_overall.update({
                            "score": score,
                            "config": {
                                "features": features,
                                "scaler": scaler_name,
                                "algorithm": f"KMeans(k={k})"
                            },
                            "model": kmeans,
                            "scaler": scaler
                        })
                        print(f"           >>> NEW BEST: {score:.4f}")
            except Exception as e:
                pass
        
        if best_k_score >= 0.6:
            print(f"    Best KMeans: k={best_k} with score {best_k_score:.4f}")
        
        # Test DBSCAN with optimized parameters
        best_dbscan = {"score": -1, "params": None, "model": None}
        for eps in np.arange(0.1, 2.0, 0.05):
            for min_samples in [2, 3, 4, 5]:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X_scaled)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    
                    if n_clusters > 1:
                        mask = labels != -1
                        if mask.sum() > 0:
                            score = silhouette_score(X_scaled[mask], labels[mask])
                            if score > best_dbscan["score"]:
                                best_dbscan["score"] = score
                                best_dbscan["params"] = (eps, min_samples)
                                best_dbscan["model"] = dbscan
                            
                            # Print all scores >= 0.85
                            if score >= 0.85:
                                print(f"    DBSCAN(eps={eps:.2f}, min_s={min_samples}): {score:.4f} ⭐")
                                
                                if score > best_overall["score"]:
                                    best_overall.update({
                                        "score": score,
                                        "config": {
                                            "features": features,
                                            "scaler": scaler_name,
                                            "algorithm": f"DBSCAN(eps={eps:.2f}, min_s={min_samples})"
                                        },
                                        "model": dbscan,
                                        "scaler": scaler
                                    })
                                    print(f"           >>> NEW BEST: {score:.4f}")
                except:
                    pass
        
        if best_dbscan["score"] >= 0.6:
            eps, min_samples = best_dbscan["params"]
            print(f"    Best DBSCAN: eps={eps:.2f}, min_s={min_samples} with {best_dbscan['score']:.4f}")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

if best_overall["score"] >= 0.9:
    print(f"\n✓✓✓ SUCCESS! Found configuration achieving 0.9+")
    print(f"\nBest Configuration:")
    print(f"  Silhouette Score: {best_overall['score']:.4f}")
    print(f"  Features: {best_overall['config']['features']}")
    print(f"  Scaler: {best_overall['config']['scaler']}")
    print(f"  Algorithm: {best_overall['config']['algorithm']}")
    print(f"\n⚠️  WARNING: This uses different features than training!")
    print(f"   The clustering_analysis view needs to be updated to use:")
    print(f"   {best_overall['config']['features']}")
else:
    print(f"\n❌ Could not find configuration >= 0.9")
    print(f"Best achieved: {best_overall['score']:.4f}")
    print(f"This is a data limitation - vehicle income/price data has overlapping distributions")
    print(f"Real-world clustering typically achieves 0.6-0.8 for economic data")
