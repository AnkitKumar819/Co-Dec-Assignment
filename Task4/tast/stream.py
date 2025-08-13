# streamlit_app.py
#type:ignore
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Load data ---
ratings_df = pd.read_csv("ratings.csv")
items_df = pd.read_csv("items.csv")
mf_data = np.load("mf_model.npz")
P, Q = mf_data["P"], mf_data["Q"]

# --- User-item matrix ---
user_item_matrix = ratings_df.pivot(index="user_id", columns="item_id", values="rating").fillna(0)
user_sim = cosine_similarity(user_item_matrix)
np.fill_diagonal(user_sim, 0)

# --- CF Recommendation ---
def recommend_cf(user_id, top_n=5):
    sim_scores = user_sim[user_id-1]
    similar_users = sim_scores.argsort()[::-1]
    
    scores = {}
    for sim_user in similar_users:
        for item, rating in enumerate(user_item_matrix.iloc[sim_user]):
            if rating > 0 and user_item_matrix.iloc[user_id-1, item] == 0:
                scores[item+1] = scores.get(item+1, 0) + sim_scores[sim_user] * rating
    ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked_items[:top_n]]

# --- MF Recommendation ---
def recommend_mf(user_id, top_n=5):
    preds = np.dot(P, Q.T)
    rated_items = user_item_matrix.iloc[user_id-1] > 0
    preds[rated_items.index[rated_items]] = -1
    top_items = preds[user_id-1].argsort()[::-1][:top_n]
    return [i+1 for i in top_items]

# --- Streamlit UI ---
st.title("📊 Recommendation System (CF & MF)")
user_id = st.number_input("Enter User ID", min_value=1, max_value=user_item_matrix.shape[0], value=1)
method = st.selectbox("Choose Method", ["Collaborative Filtering", "Matrix Factorization"])
top_n = st.slider("Top N Recommendations", 1, 10, 5)

if st.button("Get Recommendations"):
    if method == "Collaborative Filtering":
        rec_items = recommend_cf(user_id, top_n)
    else:
        rec_items = recommend_mf(user_id, top_n)
    rec_df = items_df[items_df["item_id"].isin(rec_items)]
    st.write("### Recommended Items", rec_df)
