from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load data dan encoder dari joblib (tanpa model) ===
data = joblib.load("knn_data.joblib")

train_df = data["train_df"]
df_full = data["df_full"]
le_item = data["le_item"]
le_user = data["le_user"]
le_brand = data["le_brand"]

# === Buat matriks ulang dan fit model ulang ===
user_item_matrix = train_df.pivot_table(index='NAMA_encoded', columns='KODEBARA_encoded', values='JUMLAH_UNIT', aggfunc='sum', fill_value=0)
item_user_matrix = train_df.pivot_table(index='KODEBARA_encoded', columns='NAMA_encoded', values='JUMLAH_UNIT', aggfunc='sum', fill_value=0)
brand_user_matrix = train_df.pivot_table(index='BRAND_encoded', columns='NAMA_encoded', values='JUMLAH_UNIT', aggfunc='sum', fill_value=0)

model_knn_users = NearestNeighbors(metric='cosine', algorithm='brute').fit(user_item_matrix)
model_knn_items = NearestNeighbors(metric='cosine', algorithm='brute').fit(item_user_matrix)
model_knn_brands = NearestNeighbors(metric='cosine', algorithm='brute').fit(brand_user_matrix)

# === Endpoint request body ===
class Req(BaseModel):
    selected_items: List[str]
    preference_type: str  # 'item', 'user', 'brand'

@app.post("/recommend")
def recommend_route(req: Req):
    selected_item_codes = [kode for kode in req.selected_items if kode in df_full['KODEBARA'].values]
    if not selected_item_codes:
        return {"recommendations": []}

    preference_type = req.preference_type
    encoded_items = le_item.transform(selected_item_codes)
    result_series = pd.Series(dtype='float64')

    if preference_type == 'user':
        users = train_df[train_df['KODEBARA_encoded'].isin(encoded_items)]['NAMA_encoded'].unique()
        for user in users:
            try:
                distances, indices = model_knn_users.kneighbors([user_item_matrix.loc[user]], n_neighbors=7)
                neighbors = indices.flatten()[1:]
                similar_users = user_item_matrix.iloc[neighbors]
                result_series = result_series.add(similar_users.mean(axis=0), fill_value=0)
            except:
                continue

    elif preference_type == 'item':
        for item in encoded_items:
            try:
                distances, indices = model_knn_items.kneighbors([item_user_matrix.loc[item]], n_neighbors=7)
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
        result_series = pd.Series(dtype='float64')

        for brand in selected_brands:
            try:
                distances, indices = model_knn_brands.kneighbors([brand_user_matrix.loc[brand]], n_neighbors=7)
                neighbors = indices.flatten()[1:]
                similar_brands = brand_user_matrix.iloc[neighbors].index.tolist()

                items_from_similar_brands = train_df[train_df['BRAND_encoded'].isin(similar_brands)]
                result_series = result_series.add(
                    items_from_similar_brands['KODEBARA_encoded'].value_counts(), fill_value=0
                )
            except:
                continue

        # Filter hasil hanya dari brand yang dipilih user
        allowed_items = train_df[train_df['BRAND_encoded'].isin(selected_brands)]['KODEBARA_encoded'].unique()
        result_series = result_series[result_series.index.isin(allowed_items)]

        # Fallback jika kosong
        if result_series.empty:
            result_series = train_df[train_df['BRAND_encoded'].isin(selected_brands)]['KODEBARA_encoded'].value_counts()


    # === Finalisasi hasil rekomendasi ===
    result_series = result_series.drop(labels=encoded_items, errors='ignore')
    top_items = result_series.sort_values(ascending=False).head(5).index
    recommended_codes = le_item.inverse_transform(np.array(top_items, dtype=int)).tolist()

    recommended_full = []
    for kode in recommended_codes:
        row = df_full[df_full['KODEBARA'] == kode]
        if not row.empty:
            nama = row.iloc[0]['NAMABARA']
            harga = row.iloc[0]['HARGA']
            recommended_full.append({
                "kode": str(kode),
                "nama": str(nama),
                "harga": float(harga)
            })
    print("📥 Input dari frontend:", selected_item_codes)
    print("🎯 Preferensi:", preference_type)
    print("📦 Kode rekomendasi:", recommended_codes)
    print("🔍 Detail rekomendasi:")
    for item in recommended_full:
        print(f"- {item['kode']} | {item['nama']} | Rp{item['harga']:,.0f}")

    return {"recommendations": recommended_full}

@app.get("/items")
def get_items():
    item_list = df_full[['KODEBARA', 'NAMABARA']].drop_duplicates()
    item_list['combined'] = item_list.apply(lambda row: f"{row['KODEBARA']} - {row['NAMABARA']}", axis=1)
    all_items = sorted(item_list['combined'].tolist())
    return {"items": all_items}

@app.get("/brands")
def get_brands():
    brands = sorted(df_full['BRAND'].unique().tolist())
    return {"brands": brands}
