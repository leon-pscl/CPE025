import streamlit as st
import pandas as pd

st.set_page_config(page_title="Water Quality Classifier - Log Viewer", layout="wide")

st.title("Water Quality Classifier - Log Viewer")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    # Display basic info
    st.subheader("Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Display data
    st.subheader("Data Preview")
    st.dataframe(df, use_container_width=True)
    
    # Column statistics
    st.subheader("Column Statistics")
    st.write(df.describe())

else:
    st.info("Upload CSV file to get started")
