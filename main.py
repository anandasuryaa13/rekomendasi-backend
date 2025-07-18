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
data = joblib.load("knn_data_kategori.joblib")

df_full = data["df_full"]
train_df = data["train_df"]
le_item = data["le_item"]
le_user = data["le_user"]
le_brand = data["le_brand"]

# === Matriks dan model KNN ===
user_item = train_df.pivot_table(index='USER', columns='ITEM', values='JUMLAH_UNIT', aggfunc='sum', fill_value=0)
item_user = train_df.pivot_table(index='ITEM', columns='USER', values='JUMLAH_UNIT', aggfunc='sum', fill_value=0)
brand_user = train_df.pivot_table(index='BRAND_ENC', columns='USER', values='JUMLAH_UNIT', aggfunc='sum', fill_value=0)

knn_user = NearestNeighbors(metric='cosine').fit(user_item)
knn_item = NearestNeighbors(metric='cosine').fit(item_user)
knn_brand = NearestNeighbors(metric='cosine').fit(brand_user)

# === API Request Body ===
class Req(BaseModel):
    selected_items: List[str]
    preference_type: str  # 'item', 'user', 'brand'

@app.post("/recommend")
def recommend_route(req: Req):
    selected_item_codes = [kode for kode in req.selected_items if kode in df_full['KODEBARA'].values]
    if not selected_item_codes:
        return {"recommendations": []}

    encoded_items = le_item.transform(selected_item_codes)
    result_series = pd.Series(dtype='float64')
    preference_type = req.preference_type

    if preference_type == 'user':
        users = train_df[train_df['ITEM'].isin(encoded_items)]['USER'].unique()
        for user in users:
            distances, indices = knn_user.kneighbors([user_item.loc[user]], n_neighbors=7)
            neighbors = indices.flatten()[1:]
            similar_users = user_item.iloc[neighbors]
            result_series = result_series.add(similar_users.mean(axis=0), fill_value=0)

    elif preference_type == 'item':
        for item in encoded_items:
            distances, indices = knn_item.kneighbors([item_user.loc[item]], n_neighbors=7)
            neighbors = indices.flatten()[1:]
            similar_items = item_user.iloc[neighbors]
            result_series = result_series.add(similar_items.mean(axis=0), fill_value=0)

    elif preference_type == 'brand':
        brands = train_df[train_df['ITEM'].isin(encoded_items)]['BRAND_ENC'].unique()
        for brand in brands:
            distances, indices = knn_brand.kneighbors([brand_user.loc[brand]], n_neighbors=7)
            neighbors = indices.flatten()[1:]
            similar_brand_users = brand_user.iloc[neighbors].index.tolist()
            brand_items = train_df[train_df['BRAND_ENC'].isin(similar_brand_users)]['ITEM']
            brand_items = brand_items[~brand_items.isin(encoded_items)]
            result_series = result_series.add(brand_items.value_counts(), fill_value=0)

    result_series = result_series.drop(labels=encoded_items, errors='ignore')

    # Jika kosong
    if result_series.empty:
        return {"recommendations": []}

    top_items = result_series.sort_values(ascending=False).head(5).index
    decoded = le_item.inverse_transform(top_items)

    recs = []
    for kode in decoded:
        row = df_full[df_full['KODEBARA'] == kode]
        if not row.empty:
            recs.append({
                "kode": str(kode),
                "nama": str(row.iloc[0]['NAMABARA_BERSIH']),
                "harga": float(np.float64(row.iloc[0]['HARGA']))
            })

    # Urutkan hanya 5 hasil tadi berdasarkan harga (ascending)
    recommendations = sorted(recs, key=lambda x: x['harga'])

    return {"recommendations": recommendations}

@app.get("/items")
def get_items():
    item_list = df_full[['KODEBARA', 'NAMABARA_BERSIH']].drop_duplicates()
    item_list['combined'] = item_list.apply(lambda row: f"{row['KODEBARA']} - {row['NAMABARA_BERSIH']}", axis=1)
    all_items = sorted(item_list['combined'].tolist())
    return {"items": all_items}

@app.get("/brands")
def get_brands():
    brands = sorted(df_full['BRAND'].unique().tolist())
    return {"brands": brands}

@app.get("/categories")
def get_categories():
    categories = df_full['KATEGORI'].dropna().unique().tolist()
    return {"categories": sorted(categories)}

@app.get("/kodebara-kategori")
def get_kodebara_kategori():
    mapping = df_full[['KODEBARA', 'KATEGORI']].drop_duplicates().set_index('KODEBARA')['KATEGORI'].to_dict()
    return {"map": mapping}
