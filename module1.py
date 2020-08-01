#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import altair as alt
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
st.title("ANOMALY DETECTION")
data=st.file_uploader('upload a file',type="xlsx")
if data is not None:
    df=pd.read_excel(data)
    st.write(df)
    
if data==True:
    st.df.head()
numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
def main():
    st.header("project Dataset explorer")
    st.sidebar.header("OPTIONS")
    all_cols = df.columns.values
    numeric_cols = df.select_dtypes(include=numerics).columns.values
    obj_cols = df.select_dtypes(include=["object"]).columns.values

    if st.sidebar.checkbox("Data preview", True):
        st.subheader("Data preview")
        st.markdown(f"Shape of dataset : {df.shape[0]} rows, {df.shape[1]} columns")
        if st.checkbox("Data types"):
            st.dataframe(df.dtypes)
        if st.checkbox("Data Summary"):
            st.write(df.describe())
         

    if st.sidebar.checkbox("Pattern distribution", False):
        st.subheader("Plot numeric column distribution")
        with st.echo():
            col = st.selectbox("Choose a column to display", numeric_cols)
            n_bins = st.number_input("Max number of bins ?", 5, 100, 10)
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    alt.X(f"{col}:Q", bin=alt.Bin(maxbins=n_bins)), alt.Y("count()")
                )
            )
            st.altair_chart(chart)
        st.markdown("---")

    if st.sidebar.checkbox("Scatterplot", False):
        st.subheader("Scatterplot")
        selected_cols = st.multiselect("Choose 2 columns :", numeric_cols)
        if len(selected_cols) == 2:
            color_by = st.selectbox(
                "Color by column:", all_cols, index=len(all_cols) - 1
            )
            col1, col2 = selected_cols
            chart = (
                alt.Chart(df)
                .mark_circle(size=20)
                .encode(
                    alt.X(f"{col1}:Q"), alt.Y(f"{col2}:Q"), alt.Color(f"{color_by}")
                )
                .interactive()
            )
            st.altair_chart(chart)
        st.markdown("---")


    if st.sidebar.checkbox("deviations", False):
        st.subheader("Deviation")
        with st.echo():
            selected_cols = st.multiselect(" Choose a column to display:", numeric_cols)
            if len(selected_cols) == 2:
                 color_by = st.selectbox(
                "Color by column:", all_cols, index=len(all_cols) - 1
            )
            ax = plt.subplot()
            sns.distplot(df[selected_cols][df.binary_result == 1], bins=50, label='Anormal')
            sns.distplot(df[selected_cols][df.binary_result == 0], bins=50, label='Normal')
            ax.set_xlabel('')
            ax.set_title('histogram of selected_cols: ' + str(selected_cols))
            plt.legend(loc='best')
        plt.show()
        st.markdown("---")
 


if __name__ == "__main__":
    main()


 

# In[ ]:




