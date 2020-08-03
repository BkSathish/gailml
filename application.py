#!/usr/bin/env python
# coding: utf-8

# In[8]:


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import altair as alt
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import smtplib


# In[9]:


st.title("ANOMALY DETECTION")


# In[10]:


data=st.file_uploader('upload a file',type=["csv","xlsx"])


# In[11]:

if data is not None:
    df=pd.read_excel(data)
    st.write(data)
        
      


# In[12]:

if data==True:
    st.df.head()
numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
     

# In[13]:
st.header("project Dataset explorer")
st.sidebar.header("OPTIONS") 
all_cols = df.columns.values
numeric_cols = df.select_dtypes(include=numerics).columns.values
obj_cols = df.select_dtypes(include=["object"]).columns.values
     
 


# In[14]:


if st.sidebar.checkbox("Data preview", True):
    st.subheader("Data preview")
    st.markdown(f"Shape of dataset : {df.shape[0]} rows, {df.shape[1]} columns")
    if st.checkbox("Data types"):
        st.dataframe(df.dtypes)
    if st.checkbox("Data Summary"):
        st.write(df.describe())



# seaborn plot
if st.sidebar.checkbox("Correlation plot"):
    st.subheader("Correlation plot")
    cor = df.corr()
    mask = np.zeros_like(cor)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(12,10))
    with sns.axes_style("white"):
        st.write(sns.heatmap(cor,annot=True,linewidth=2,mask = mask,cmap="magma"))
        st.pyplot()




# In[24]:


if st.sidebar.checkbox("Deviations"):
    st.subheader("Deviation plot")  
    for feature in ['time', 'measurement','control_mode','binary_result']:
        ax = plt.subplot()
        st.write(sns.distplot(df[feature][df.binary_result == 1], bins=50, label='Anormal',kde_kws={'bw':0.02}))
        st.write(sns.distplot(df[feature][df.binary_result == 0], bins=50, label='Normal',kde_kws={'bw':0.02}))
        ax.set_xlabel('')
        ax.set_title('histogram of feature: ' + str(feature))
        plt.legend(loc='best')
        st.pyplot()


# In[25]:


def ztest(feature):
    mean = falsepositive[feature].mean()
    std = falsepositive[feature].std()
    zScore = (falsenegative[feature].mean() - mean) / (std/np.sqrt(sample_size))
    return zScore     

columns= df.drop('binary_result', axis=1).columns
falsepositive= df[df.binary_result==0]
falsenegative= df[df.binary_result==1]
sample_size=len(falsepositive)
significant_features=[]
setpoint=70
for i in columns:
    z_value=ztest(i)
    if( abs(z_value) >= setpoint):    
       
        significant_features.append(i)


# In[26]:


    
significant_features.append('binary_result')
y= df[significant_features]
inliers = df[df.binary_result==0]
ins = inliers.drop(['binary_result'], axis=1)
outliers = df[df.binary_result==1]
outs = outliers.drop(['binary_result'], axis=1)

        


# In[27]:


def falsepositive_accuracy(values):
    tp=list(values).count(1)
    total=values.shape[0]
    accuracy=np.round(tp/total,4)
    return accuracy


# In[28]:


def falsenegative_accuracy(values):
    tn=list(values).count(-1)
    total=values.shape[0]
    accuracy=np.round(tn/total,4)
    return accuracy


# In[29]:


st.subheader("Accuracy of Alarms")
ISF = IsolationForest(random_state=42)
ISF.fit(ins)
falsepositive_isf = ISF.predict(ins)
falsenegative_isf = ISF.predict(outs)
in_accuracy_isf=falsepositive_accuracy(falsepositive_isf)
out_accuracy_isf=falsenegative_accuracy(falsenegative_isf)
st.write("Accuracy in Detecting falsepositive Alarm:", in_accuracy_isf)
st.write("Accuracy in Detecting falsenegative Alarm:", out_accuracy_isf)


# In[30]:





# In[31]:


if st.sidebar.checkbox("Alarm Report", False):
    st.subheader("classification of Alarm")
    fig, (ax1,ax2)= plt.subplots(1,2,figsize=[16,3])
    ax1.set_title("Accuracy of Isolation Forest",fontsize=20)
    st.write(sns.barplot(x=[in_accuracy_isf,out_accuracy_isf], y=['falsepositive Alarm', 'falsenegative Alarm'], label="classifiers",  color="b", ax=ax1))
    ax1.set(xlim=(0,1))
   
    st.pyplot()


# In[ ]:

if st.sidebar.checkbox("Email", False):
    st.subheader("Email")
    email_sender=st.text_input("Enter User Email--")
    password=st.text_input("Enter User password--",type="password")
    email_reciever=st.text_input("Enter Reciever Email--")
    subject=st.text_input(" Email subject")
    body=("Accuracy in Detecting falsepositive Alarm:", in_accuracy_isf,
    "Accuracy in Detecting falsenegative Alarm:", out_accuracy_isf)
    if st.button("Send Email"):
        server=smtplib.SMTP("smtp.gmail.com",587)
        server.starttls()
        server.login(email_sender,password)
        message="subject:{}\n\n{}".format(subject,body)
        server.sendmail(email_sender,email_reciever,message)
        server.quit()
        st.success("Email Send successfully.")


