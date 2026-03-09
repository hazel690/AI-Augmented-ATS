

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch


st.set_page_config(page_title="AI Talent Scout Pro", layout="wide")

st.title(" AI-Augmented Recruitment Engine")
st.markdown("### Search 10,000+ Candidates using Semantic Intelligence")


@st.cache_resource
def load_assets():
    df = pd.read_csv('master_hr_data.csv')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    embeddings = model.encode(df['Resume_Content'].tolist(), convert_to_tensor=True)
    return df, model, embeddings

df, model, resume_embeddings = load_assets()


st.sidebar.header("Filter Settings")
min_exp = st.sidebar.slider("Minimum Years of Experience", 0, 15, 0)
num_results = st.sidebar.number_input("Candidates to Display", 1, 20, 5)


jd_input = st.text_area("Paste Job Description (JD) or Search Query here:", 
                         placeholder="e.g., Looking for a People Analyst with SQL and Python skills...",
                         height=200)

if st.button("Find Top Talent"):
    if jd_input:
    
        query_embedding = model.encode(jd_input, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, resume_embeddings)[0]
        df['Match_Score'] = scores.tolist()
        
        
        results = df[df['Years_Exp'] >= min_exp].sort_values(by='Match_Score', ascending=False).head(num_results)
        
        
        st.success(f"Found {len(results)} matches!")
        for _, row in results.iterrows():
            with st.expander(f" {row['Name']} - Match Score: {row['Match_Score']:.2f}"):
                st.write(f"**Role:** {row['Applied_Role']} | **Experience:** {row['Years_Exp']} Years")
                st.write(f"**Resume Snippet:** {row['Resume_Content']}")
                st.progress(row['Match_Score'])
    else:
        st.warning("Please enter a JD to start the search.")
