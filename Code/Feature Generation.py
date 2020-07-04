#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("Dataset.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df['Class'].value_counts()


# In[6]:


len(df.iloc[1,0])


# In[7]:


single_neucliotides = ['A', 'C', 'G', 'U']

for s_neucliotide in single_neucliotides:
    for i in range(41):
        feature_name = 'Index' + str(i) + '_' + s_neucliotide
        values = []
        for j in range(df.shape[0]):
            sequence = df.iloc[j,0]
            if sequence[i] == s_neucliotide:
                values.append(1)
            else:
                values.append(0)
        df[feature_name] = values


# In[8]:


df.head()


# In[9]:


di_neucliotides = ["AA", "AC", "AG", "AU", "CA", "CC", "CG", "CU", "GA", "GC", "GG", "GU", "UA", "UC", "UG", "UU"]

for di_neucliotide in di_neucliotides:
    for i in range(41-1):
        feature_name = 'Index' + str(i) + "," + str(i+1) + '_' + di_neucliotide
        values = []
        neucliotides = list(di_neucliotide)
        for j in range(df.shape[0]):
            sequence = df.iloc[j,0]
            if sequence[i] == neucliotides[0] and sequence[i+1] == neucliotides[1]:
                values.append(1)
            else:
                values.append(0)
        df[feature_name] = values


# In[10]:


df.head()


# In[11]:


tri_neucliotides = ["AAA", "AAC", "AAG", "AAU", "ACA", "ACC", "ACG", "ACU", "AGA", "AGC", "AGG", "AGU", "AUA", "AUC", "AUG", "AUU", "CAA", 
    "CAC", "CAG", "CAU", "CCA", "CCC", "CCG", "CCU", "CGA", "CGC","CGG", "CGU", "CUA", "CUC", "CUG", "CUU", "GAA", "GAC", "GAG",
    "GAU", "GCA", "GCC", "GCG", "GCU", "GGA", "GGC", "GGG", "GGU", "GUA", "GUC", "GUG", "GUU", "UAA", "UAC", "UAG", "UAU",
    "UCA", "UCC", "UCG", "UCU", "UGA", "UGC", "UGG", "UGU", "UUA", "UUC", "UUG", "UUU"]

for tri_neucliotide in tri_neucliotides:
    for i in range(41-2):
        feature_name = 'Index' + str(i) + "," + str(i+1) + "," + str(i+2) + '_' + tri_neucliotide
        values = []
        neucliotides = list(tri_neucliotide)
        for j in range(df.shape[0]):
            sequence = df.iloc[j,0]
            if sequence[i] == neucliotides[0] and sequence[i+1] == neucliotides[1] and sequence[i+2] == neucliotides[2]:
                values.append(1)
            else:
                values.append(0)
        df[feature_name] = values


# In[12]:


df.head()


# In[19]:


tri_neucliotides = ["AAA", "AAC", "AAG", "AAU", "ACA", "ACC", "ACG", "ACU", "AGA", "AGC", "AGG", "AGU", "AUA", "AUC", "AUG", "AUU", "CAA", 
    "CAC", "CAG", "CAU", "CCA", "CCC", "CCG", "CCU", "CGA", "CGC","CGG", "CGU", "CUA", "CUC", "CUG", "CUU", "GAA", "GAC", "GAG",
    "GAU", "GCA", "GCC", "GCG", "GCU", "GGA", "GGC", "GGG", "GGU", "GUA", "GUC", "GUG", "GUU", "UAA", "UAC", "UAG", "UAU",
    "UCA", "UCC", "UCG", "UCU", "UGA", "UGC", "UGG", "UGU", "UUA", "UUC", "UUG", "UUU"]

for tri_neucliotide in tri_neucliotides:
    for i in range(41-3):
        feature_name = 'Index' + str(i) + "," + str(i+1) + "," + str(i+2) + "," + str(i+3) + '_' + tri_neucliotide[0] + '-' + tri_neucliotide[1:]
        values = []
        neucliotides = list(tri_neucliotide)
        for j in range(df.shape[0]):
            sequence = df.iloc[j,0]
            if sequence[i] == neucliotides[0] and sequence[i+2] == neucliotides[1] and sequence[i+3] == neucliotides[2]:
                values.append(1)
            else:
                values.append(0)
        df[feature_name] = values


# In[20]:


df.head()


# In[21]:


tri_neucliotides = ["AAA", "AAC", "AAG", "AAU", "ACA", "ACC", "ACG", "ACU", "AGA", "AGC", "AGG", "AGU", "AUA", "AUC", "AUG", "AUU", "CAA", 
    "CAC", "CAG", "CAU", "CCA", "CCC", "CCG", "CCU", "CGA", "CGC","CGG", "CGU", "CUA", "CUC", "CUG", "CUU", "GAA", "GAC", "GAG",
    "GAU", "GCA", "GCC", "GCG", "GCU", "GGA", "GGC", "GGG", "GGU", "GUA", "GUC", "GUG", "GUU", "UAA", "UAC", "UAG", "UAU",
    "UCA", "UCC", "UCG", "UCU", "UGA", "UGC", "UGG", "UGU", "UUA", "UUC", "UUG", "UUU"]

for tri_neucliotide in tri_neucliotides:
    for i in range(41-4):
        feature_name = 'Index' + str(i) + "," + str(i+1) + "," + str(i+2) + "," + str(i+3) + "," + str(i+4) + '_' + tri_neucliotide[0] + '--' + tri_neucliotide[1:]
        values = []
        neucliotides = list(tri_neucliotide)
        for j in range(df.shape[0]):
            sequence = df.iloc[j,0]
            if sequence[i] == neucliotides[0] and sequence[i+3] == neucliotides[1] and sequence[i+4] == neucliotides[2]:
                values.append(1)
            else:
                values.append(0)
        df[feature_name] = values


# In[22]:


df.head()


# In[23]:


tri_neucliotides = ["AAA", "AAC", "AAG", "AAU", "ACA", "ACC", "ACG", "ACU", "AGA", "AGC", "AGG", "AGU", "AUA", "AUC", "AUG", "AUU", "CAA", 
    "CAC", "CAG", "CAU", "CCA", "CCC", "CCG", "CCU", "CGA", "CGC","CGG", "CGU", "CUA", "CUC", "CUG", "CUU", "GAA", "GAC", "GAG",
    "GAU", "GCA", "GCC", "GCG", "GCU", "GGA", "GGC", "GGG", "GGU", "GUA", "GUC", "GUG", "GUU", "UAA", "UAC", "UAG", "UAU",
    "UCA", "UCC", "UCG", "UCU", "UGA", "UGC", "UGG", "UGU", "UUA", "UUC", "UUG", "UUU"]

for tri_neucliotide in tri_neucliotides:
    for i in range(41-5):
        feature_name = 'Index' + str(i) + "," + str(i+1) + "," + str(i+2) + "," + str(i+3) + "," + str(i+4) + "," + str(i+5) + '_' + tri_neucliotide[0] + '---' + tri_neucliotide[1:]
        values = []
        neucliotides = list(tri_neucliotide)
        for j in range(df.shape[0]):
            sequence = df.iloc[j,0]
            if sequence[i] == neucliotides[0] and sequence[i+4] == neucliotides[1] and sequence[i+5] == neucliotides[2]:
                values.append(1)
            else:
                values.append(0)
        df[feature_name] = values


# In[24]:


df.head()


# In[25]:


tri_neucliotides = ["AAA", "AAC", "AAG", "AAU", "ACA", "ACC", "ACG", "ACU", "AGA", "AGC", "AGG", "AGU", "AUA", "AUC", "AUG", "AUU", "CAA", 
    "CAC", "CAG", "CAU", "CCA", "CCC", "CCG", "CCU", "CGA", "CGC","CGG", "CGU", "CUA", "CUC", "CUG", "CUU", "GAA", "GAC", "GAG",
    "GAU", "GCA", "GCC", "GCG", "GCU", "GGA", "GGC", "GGG", "GGU", "GUA", "GUC", "GUG", "GUU", "UAA", "UAC", "UAG", "UAU",
    "UCA", "UCC", "UCG", "UCU", "UGA", "UGC", "UGG", "UGU", "UUA", "UUC", "UUG", "UUU"]

for tri_neucliotide in tri_neucliotides:
    for i in range(41-3):
        feature_name = 'Index' + str(i) + "," + str(i+1) + "," + str(i+2) + "," + str(i+3) + '_' + tri_neucliotide[:2] + '-' + tri_neucliotide[2]
        values = []
        neucliotides = list(tri_neucliotide)
        for j in range(df.shape[0]):
            sequence = df.iloc[j,0]
            if sequence[i] == neucliotides[0] and sequence[i+1] == neucliotides[1] and sequence[i+3] == neucliotides[2]:
                values.append(1)
            else:
                values.append(0)
        df[feature_name] = values


# In[26]:


df.head()


# In[27]:


tri_neucliotides = ["AAA", "AAC", "AAG", "AAU", "ACA", "ACC", "ACG", "ACU", "AGA", "AGC", "AGG", "AGU", "AUA", "AUC", "AUG", "AUU", "CAA", 
    "CAC", "CAG", "CAU", "CCA", "CCC", "CCG", "CCU", "CGA", "CGC","CGG", "CGU", "CUA", "CUC", "CUG", "CUU", "GAA", "GAC", "GAG",
    "GAU", "GCA", "GCC", "GCG", "GCU", "GGA", "GGC", "GGG", "GGU", "GUA", "GUC", "GUG", "GUU", "UAA", "UAC", "UAG", "UAU",
    "UCA", "UCC", "UCG", "UCU", "UGA", "UGC", "UGG", "UGU", "UUA", "UUC", "UUG", "UUU"]

for tri_neucliotide in tri_neucliotides:
    for i in range(41-4):
        feature_name = 'Index' + str(i) + "," + str(i+1) + "," + str(i+2) + "," + str(i+3) + '_' + tri_neucliotide[:2] + '--' + tri_neucliotide[2]
        values = []
        neucliotides = list(tri_neucliotide)
        for j in range(df.shape[0]):
            sequence = df.iloc[j,0]
            if sequence[i] == neucliotides[0] and sequence[i+1] == neucliotides[1] and sequence[i+4] == neucliotides[2]:
                values.append(1)
            else:
                values.append(0)
        df[feature_name] = values


# In[28]:


df.head()


# In[29]:


tri_neucliotides = ["AAA", "AAC", "AAG", "AAU", "ACA", "ACC", "ACG", "ACU", "AGA", "AGC", "AGG", "AGU", "AUA", "AUC", "AUG", "AUU", "CAA", 
    "CAC", "CAG", "CAU", "CCA", "CCC", "CCG", "CCU", "CGA", "CGC","CGG", "CGU", "CUA", "CUC", "CUG", "CUU", "GAA", "GAC", "GAG",
    "GAU", "GCA", "GCC", "GCG", "GCU", "GGA", "GGC", "GGG", "GGU", "GUA", "GUC", "GUG", "GUU", "UAA", "UAC", "UAG", "UAU",
    "UCA", "UCC", "UCG", "UCU", "UGA", "UGC", "UGG", "UGU", "UUA", "UUC", "UUG", "UUU"]

for tri_neucliotide in tri_neucliotides:
    for i in range(41-5):
        feature_name = 'Index' + str(i) + "," + str(i+1) + "," + str(i+2) + "," + str(i+3) + '_' + tri_neucliotide[:2] + '---' + tri_neucliotide[2]
        values = []
        neucliotides = list(tri_neucliotide)
        for j in range(df.shape[0]):
            sequence = df.iloc[j,0]
            if sequence[i] == neucliotides[0] and sequence[i+1] == neucliotides[1] and sequence[i+5] == neucliotides[2]:
                values.append(1)
            else:
                values.append(0)
        df[feature_name] = values


# In[30]:


df.head()


# In[31]:


tri_neucliotides = ["AAA", "AAC", "AAG", "AAU", "ACA", "ACC", "ACG", "ACU", "AGA", "AGC", "AGG", "AGU", "AUA", "AUC", "AUG", "AUU", "CAA", 
    "CAC", "CAG", "CAU", "CCA", "CCC", "CCG", "CCU", "CGA", "CGC","CGG", "CGU", "CUA", "CUC", "CUG", "CUU", "GAA", "GAC", "GAG",
    "GAU", "GCA", "GCC", "GCG", "GCU", "GGA", "GGC", "GGG", "GGU", "GUA", "GUC", "GUG", "GUU", "UAA", "UAC", "UAG", "UAU",
    "UCA", "UCC", "UCG", "UCU", "UGA", "UGC", "UGG", "UGU", "UUA", "UUC", "UUG", "UUU"]

for tri_neucliotide in tri_neucliotides:
    for i in range(41-4):
        feature_name = 'Index' + str(i) + "," + str(i+1) + "," + str(i+2) + "," + str(i+3) + '_' + tri_neucliotide[0] + '-' + tri_neucliotide[1] + '-' + tri_neucliotide[2]
        values = []
        neucliotides = list(tri_neucliotide)
        for j in range(df.shape[0]):
            sequence = df.iloc[j,0]
            if sequence[i] == neucliotides[0] and sequence[i+2] == neucliotides[1] and sequence[i+4] == neucliotides[2]:
                values.append(1)
            else:
                values.append(0)
        df[feature_name] = values


# In[32]:


df.head()


# In[33]:


feature_name = 'GC_Content'
values = []
for j in range(df.shape[0]):
    sequence = df.iloc[j,0]
    A_count = sequence.count("A")
    C_count = sequence.count("C")
    G_count = sequence.count("G")
    U_count = sequence.count("U")
    ratio = round((G_count + C_count)/(A_count + G_count + C_count + U_count),2)
    values.append(ratio)
df[feature_name] = values
df.head()


# In[34]:


new_df = df.iloc[:,1:]
new_df.head()


# In[35]:


new_df.to_csv("Features_Dataset.csv",index=False)


# In[ ]:




