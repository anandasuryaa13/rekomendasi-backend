import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, confusion_matrix

# === Load data dan encoder dari joblib ===
data = joblib.load("knn_data.joblib")
train_df = data["train_df"]
test_df = data["test_df"]  # ✅ Gunakan test_df untuk evaluasi
df = data["df_full"]
le_item = data["le_item"]
le_user = data["le_user"]
le_brand = data["le_brand"]

# === Buat ulang matriks dan model KNN ===
user_item_matrix = train_df.pivot_table(index='NAMA_encoded', columns='KODEBARA_encoded', values='JUMLAH_UNIT', aggfunc='sum', fill_value=0)
item_user_matrix = train_df.pivot_table(index='KODEBARA_encoded', columns='NAMA_encoded', values='JUMLAH_UNIT', aggfunc='sum', fill_value=0)
brand_user_matrix = train_df.pivot_table(index='BRAND_encoded', columns='NAMA_encoded', values='JUMLAH_UNIT', aggfunc='sum', fill_value=0)

model_knn_users = NearestNeighbors(metric='cosine', algorithm='brute').fit(user_item_matrix)
model_knn_items = NearestNeighbors(metric='cosine', algorithm='brute').fit(item_user_matrix)
model_knn_brands = NearestNeighbors(metric='cosine', algorithm='brute').fit(brand_user_matrix)

# === Fungsi rekomendasi lokal ===
def recommend_knn(selected_item_codes, preference_type='item', n=5):
    result_series = pd.Series(dtype='float64')
    encoded_items = le_item.transform(selected_item_codes)

    if preference_type == 'user':
        users = train_df[train_df['KODEBARA_encoded'].isin(encoded_items)]['NAMA_encoded'].unique()
        for user in users:
            try:
                distances, indices = model_knn_users.kneighbors([user_item_matrix.loc[user]], n_neighbors=6)
                neighbors = indices.flatten()[1:]
                similar_users = user_item_matrix.iloc[neighbors]
                result_series = result_series.add(similar_users.mean(axis=0), fill_value=0)
            except:
                continue

    elif preference_type == 'item':
        for item in encoded_items:
            try:
                distances, indices = model_knn_items.kneighbors([item_user_matrix.loc[item]], n_neighbors=6)
                neighbors = indices.flatten()[1:]
                similar_items = item_user_matrix.iloc[neighbors]
                result_series = result_series.add(similar_items.mean(axis=0), fill_value=0)
            except:
                continue

        selected_categories = train_df[train_df['KODEBARA_encoded'].isin(encoded_items)]['NAMABARA'].str.split().str[1].unique()
        selected_brands = train_df[train_df['KODEBARA_encoded'].isin(encoded_items)]['BRAND_encoded'].unique()

        filtered_items = train_df[
            (train_df['NAMABARA'].str.split().str[1].isin(selected_categories)) &
            (~train_df['BRAND_encoded'].isin(selected_brands))
        ]['KODEBARA_encoded'].unique()

        result_series = result_series[result_series.index.isin(filtered_items)]

    elif preference_type == 'brand':
        selected_brands = train_df[train_df['KODEBARA_encoded'].isin(encoded_items)]['BRAND_encoded'].unique()
        for brand in selected_brands:
            try:
                distances, indices = model_knn_brands.kneighbors([brand_user_matrix.loc[brand]], n_neighbors=6)
                neighbors = indices.flatten()[1:]
                similar_brands = brand_user_matrix.iloc[neighbors].index.tolist()
                items_from_similar_brands = train_df[train_df['BRAND_encoded'].isin(similar_brands)]
                result_series = result_series.add(items_from_similar_brands['KODEBARA_encoded'].value_counts(), fill_value=0)
            except:
                continue

        filtered_items = train_df[train_df['BRAND_encoded'].isin(selected_brands)]['KODEBARA_encoded'].unique()
        result_series = result_series[result_series.index.isin(filtered_items)]

        if result_series.empty:
            result_series = train_df[train_df['BRAND_encoded'].isin(selected_brands)]['KODEBARA_encoded'].value_counts()

    result_series = result_series.drop(labels=encoded_items, errors='ignore')
    top_items = result_series.sort_values(ascending=False).head(n).index
    decoded_items = le_item.inverse_transform(np.array(top_items, dtype=int)).tolist()

    return decoded_items

# === Fungsi evaluasi ===
def evaluate_recommendation(recommended_items, test_df, df_full, n=5):
    ground_truth_items = test_df['KODEBARA'].unique()
    recommended_items = recommended_items[:n]

    true_positives = len([item for item in recommended_items if item in ground_truth_items])
    precision_at_n = true_positives / n if n > 0 else 0
    recall_at_n = true_positives / len(ground_truth_items) if len(ground_truth_items) > 0 else 0
    hit_rate = 1 if true_positives > 0 else 0
    coverage = len(set(recommended_items)) / df_full['KODEBARA'].nunique()

    print(f"\n📈 Evaluation Metrics for Top-{n}")
    print(f"Precision@{n}: {precision_at_n:.2f}")
    print(f"Recall@{n}: {recall_at_n:.2f}")
    print(f"Hit Rate: {'Yes' if hit_rate else 'No'}")
    print(f"Coverage: {coverage:.2%}")

    all_items = df_full['KODEBARA'].unique()
    y_true = [1 if item in ground_truth_items else 0 for item in all_items]
    y_pred = [1 if item in recommended_items else 0 for item in all_items]

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("📉 Confusion Matrix:")
    print(cm)

    return {
        'precision_at_n': precision_at_n,
        'recall_at_n': recall_at_n,
        'hit_rate': hit_rate,
        'coverage': coverage,
        'confusion_matrix': cm
    }

# === Fungsi untuk panggil backend ===
def fetch_backend_recommendation(selected_items, preference):
    url = "http://127.0.0.1:8000/recommend"
    payload = {
        "selected_items": selected_items,
        "preference_type": preference
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return [item['kode'] for item in response.json()['recommendations']]
    else:
        print(f"❌ Error {response.status_code} from backend.")
        return []

# === Input untuk dites ===
selected_items = ['BLF-001', 'BLF-004']

# === Loop untuk semua preferensi ===
for pref in ['user', 'item', 'brand']:
    print(f"\n=== 🔍 Membandingkan preferensi: {pref.upper()} ===")

    local_reco = recommend_knn(selected_items, preference_type=pref)
    backend_reco = fetch_backend_recommendation(selected_items, pref)

    print("📘 Lokal  :", local_reco)
    print("🖥️ Backend:", backend_reco)

    if local_reco == backend_reco:
        print("✅ HASIL SAMA")
    else:
        beda = set(local_reco).symmetric_difference(set(backend_reco))
        print("❌ HASIL BEDA, perbedaan kode:", beda)

    print("\n📘 Evaluasi rekomendasi lokal:")
    evaluate_recommendation(local_reco, test_df=test_df, df_full=df)
