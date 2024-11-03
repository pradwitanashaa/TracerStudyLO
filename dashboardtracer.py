import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import urllib
import re
import nltk
import altair as alt
import itertools
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

## TOP COMPETENCIES
# database cos-miniLM
data = pd.read_csv("https://docs.google.com/spreadsheets/d/1pD90XJBNH4I8Jkzrx9PdWA26-asGWBJRhahVSCHAPEk/export?format=csv&gid=0")
data['Competencies'] = data['Competencies'].replace({'Memecahkan Masalah Komplekss': 'Memecahkan Masalah Kompleks'})
data['Jurusan'] = data['Jurusan'].replace({'Manajemen Rekayasa Industri': 'Manajemen Rekayasa'})
listprodi = data['Jurusan'].unique()
competencies = data["Competencies"].unique()
data['dist'] = data['dist'].str.replace(',', '.')
data['dist'] = pd.to_numeric(data['dist'])

data['normalized_dist'] = data['normalized_dist'].str.replace(',', '.')
data['normalized_dist'] = pd.to_numeric(data['normalized_dist'])

translate = {}
data_translate = pd.read_csv("https://docs.google.com/spreadsheets/d/1SlzxU-xPxfU5-sMbJpogpQEmXb4ntX-gi4kqCAcqh0o/export?format=csv&gid=0")
for i in range(len(data_translate)):
  row = data_translate.iloc[i]
  translate[row["Inggris"]] = row["Indo"]

data['LO'] = data['LO'].replace(translate)

# NGAMBIL JURUSAN

buatbaru = []
buattop3 = []

for jurusan in listprodi:
  # NGAMBIL LO di Jurusan
  los = data[data['Jurusan'] == jurusan]["LO"].unique()
  for lo in los:
    data2 = data[(data['Jurusan'] == jurusan) & (data["LO"] == lo)].sort_values('dist',ascending=False)
    buatbaru.append(data2)
    buattop3.append(data2.iloc[:3])

baru = pd.concat(buatbaru)
top3 = pd.concat(buattop3)

# Buat nyari top three Copmetencies untuk setiap LO di jurusan input
def top_three(jurusan):
  ret = []
  los = data[data['Jurusan'] == jurusan]["LO"].unique()
  for lo in los:
    data2 = data[(data['Jurusan'] == jurusan) & (data["LO"] == lo)].sort_values('dist',ascending=False)
    ret.append(data2.iloc[:3])
  return pd.concat(ret)

top3results = []
for jurusan in listprodi:
  top3results.append(top_three(jurusan))
topthree = pd.concat(top3results)

## Cos Similarities (Weizsfield)
# Defining the scaling function
def scaling(p):
  p_scaling = []
  for i in p:
    x = (i-1)/(5-1)
    p_scaling.append(x)
  return p_scaling

# Defining the deleting zero function
def delo(vector):
  p = np.array(vector)
  p = np.delete(p, np.where(p == 0))
  return p

# Defining the weight function
def weight(distance):
  distance = np.array(distance)
  # Handling division by zero
  if 0 in distance:
    distance1 = delo(distance)
    dbar = min(distance1)
    idx0 = np.where(distance == 0)[0]
    for i in idx0:
      distance[i] = dbar
  # Getting weight values
  weight = [1/d for d in distance]
  return weight

# Defining Weiszfeld algorithm function
def weiszfeld(p1,T):
  error = 0.00001

  if T == 1: # Scaled
    p = p1
  else:      # Not scaled
    p = scaling(p1)

  x_bar = sum(p)/len(p)
  distance = [abs(x_bar-x) for x in p]
  weight1 = weight(distance)
  x_new = sum([x*w for x,w in zip(p,weight1)]) / sum(weight1)
  while abs(x_new-x_bar) > error:
    x_bar = x_new
    distance = [abs(x_bar-x) for x in p]
    weight1 = weight(distance)
    x_new = sum([x*w for x,w in zip(p,weight1)]) / sum(weight1)
  return x_new

# Defining function to find value of competency
# Using weisfeld algorithm
def value_competency(prodi,df):
  df = df[df['Jurusan']== prodi]

  # Using weiszfeld algorithm to get representative of value of competency
  vc = df.iloc[:,2:25].values
  vc = np.transpose(vc)

  rvc = [] # representative of value of competency using weiszfeld algorithm
  for i in range(23):
    v = weiszfeld(vc[i],0)
    rvc.append(v)
  return rvc

import gdown

# Load data
file_id = '1ZayQZrql2cH2-oAezHIWZB1w9u2qK2Xj'
# Construct the download URL
url = f'https://drive.google.com/uc?id={file_id}'
output = 'competency_all.xlsx'
gdown.download(url, output, quiet=False)

# Load the downloaded Excel file
df = pd.read_excel(output, sheet_name=str("Copy of 17"))
# https://docs.google.com/spreadsheets/d/1ZayQZrql2cH2-oAezHIWZB1w9u2qK2Xj/edit?gid=456732573#gid=456732573

# Change the column name
df.columns.values[1] = 'Jurusan'
listprodi = df['Jurusan'].unique()

# Membuat kombinasi Cartesian antara program studi dan kompetensi
combinations = list(itertools.product(listprodi, competencies))

# Membuat dataframe
all = pd.DataFrame(combinations, columns=['Jurusan', 'Competencies'])

all['Weiszfeild'] = ''

for prodi in listprodi:
  all.loc[all['Jurusan'] == prodi, 'Weiszfeild'] = value_competency(prodi, df)
merged_df = pd.merge(data, all, on=['Jurusan', 'Competencies'], how='inner')

def cos_similarity(jurusan):
  selected = merged_df[merged_df["Jurusan"]==jurusan]
  score = {}
  for LO in selected['LO']:
    selected_LO = selected[selected["LO"]==LO]
    dot = np.dot(selected_LO['dist'],selected_LO['Weiszfeild'])
    norm_dist = np.dot(selected_LO['dist'],selected_LO['dist'])
    norm_weisz = np.dot(selected_LO['Weiszfeild'],selected_LO['Weiszfeild'])
    cos = dot/((norm_dist*norm_weisz)**0.5)
    score[LO] = cos
  return score
results = []
for jurusan in listprodi:
    # cari fakultas dari 'jurusan' di sini
    cos_scores = cos_similarity(jurusan)
    # Append results to list with format [(jurusan, LO, cos_score)]
    for LO, score in cos_scores.items():
        results.append((jurusan, LO, score))

datacss = pd.DataFrame(results, columns=['Jurusan', 'LO', 'Cosine Similarity'])
database = pd.merge(data, datacss, on=['Jurusan','LO'], how='inner')
database = database.drop_duplicates()

st.title('Visualisasi Kompetensi dan LO')
st.header('Tracer Study ITB')
tab1, tab2 = st.tabs(["Analisis per Fakultas", "Analisis per Program Studi"])
with tab1:
  st.write(
    """
    # Visualisasi 23 Kompetensi
    Kompetensi berikut dihitung berdasarkan tiga besar kompetensi yang paling relevan dengan setiap Learning Outcomes program studi.
    """
  )
  options = ['Pilih fakultas...'] + list(topthree['Fakultas'].unique())
  fakultas = st.selectbox("Fakultas",options)
  topkompetensi = topthree[topthree['Fakultas'] == fakultas]['Competencies'].value_counts()
  topkompetensi_sort = topkompetensi.sort_values(ascending=False)
  # topkompetensi.plot(kind="barh")
  val = list(topkompetensi_sort)
  index = list(topkompetensi_sort.index)

  spaces = []
  index2 = []
  for idx,lo in enumerate(index):
    k = len(lo)
    spaces = []
    for i in range(k):
      if lo[i] == " ":
        spaces.append(i)

    for i in range(len(spaces)-1,-1,-4):
      lo = lo[:spaces[i]] + "\n" + lo[spaces[i]+1:]
    index2.append(lo)

  # Create a DataFrame with the data
  plottop3 = pd.DataFrame({
    'Kompetensi': index2,
    'Frekuensi': val
    })

  # Set 'Kompetensi' as index for proper bar plotting
  plottop3 = plottop3.set_index('Kompetensi')

  # Plot Cos Similarities
  datagacor = database[database['Fakultas'] == fakultas].sort_values('Cosine Similarity',ascending=True)
  los = list(datagacor['LO'])
  css = list(datagacor['Cosine Similarity'])
  los2 = []
  for idx,lo in enumerate(los):
    k = len(lo)
    spaces = []
    for i in range(k):
      if lo[i] == " ":
        spaces.append(i)

    for i in range(len(spaces)-1,-1,-6):
      lo = lo[:spaces[i]] + "\n" + lo[spaces[i]+1:]
    los2.append(lo)
    
  plotcss = pd.DataFrame({
    'LO': los2,
    'Cosine Similarity': css
    })

  # Set 'Kompetensi' as index for proper bar plotting
  plotcss = plotcss.set_index('LO')

  # Make sure plottop3 is sorted
  plottop3 = plottop3.sort_values(by='Frekuensi', ascending=False)

  # Create an Altair bar chart
  chart1 = alt.Chart(plottop3.reset_index()).mark_bar().encode(
    x=alt.X('Frekuensi:Q', title="Frekuensi"),
    y=alt.Y('Kompetensi:N', sort='-x', title="Kompetensi")
    ).properties(width=1000, height=800)

  st.altair_chart(chart1, use_container_width=True)

  # Plot Dashboard
  # st.bar_chart(plottop3,x_label="Frekuensi",y_label="Kompetensi",
  #              horizontal=True,use_container_width=True)
  st.dataframe(data=plottop3)

with tab2:
  st.write(
    """
    # Visualisasi 23 Kompetensi
    Kompetensi berikut dihitung berdasarkan tiga besar kompetensi yang paling relevan dengan setiap Learning Outcomes (LO) program studi.
    """
  )
  options = ['Pilih program studi...'] + list(topthree['Jurusan'].unique())
  prodi = st.selectbox("Program Studi",options)
  topkompetensi = topthree[topthree['Jurusan'] == prodi]['Competencies'].value_counts()
  topkompetensi_sort = topkompetensi.sort_values(ascending=False)
  # topkompetensi.plot(kind="barh")
  val = list(topkompetensi_sort)
  index = list(topkompetensi_sort.index)

  spaces = []
  index2 = []
  for idx,lo in enumerate(index):
    k = len(lo)
    spaces = []
    for i in range(k):
      if lo[i] == " ":
        spaces.append(i)

    for i in range(len(spaces)-1,-1,-4):
      lo = lo[:spaces[i]] + "\n" + lo[spaces[i]+1:]
    index2.append(lo)

  # Create a DataFrame with the data
  plottop3 = pd.DataFrame({
    'Kompetensi': index2,
    'Frekuensi': val
    })

  # Set 'Kompetensi' as index for proper bar plotting
  plottop3 = plottop3.set_index('Kompetensi')

  # Plot Cos Similarities
  datagacor = database[database['Jurusan'] == prodi].sort_values('Cosine Similarity',ascending=False)
  los = list(datagacor['LO'])
  css = list(datagacor['Cosine Similarity'])
  los2 = []
  for idx,lo in enumerate(los):
    k = len(lo)
    spaces = []
    for i in range(k):
      if lo[i] == " ":
        spaces.append(i)

    for i in range(len(spaces)-1,-1,-6):
      lo = lo[:spaces[i]] + "\n" + lo[spaces[i]+1:]
    los2.append(lo)
    
  plotcss = pd.DataFrame({
    'LO': los2,
    'Cosine Similarity': css
    })
  
  plotcss = plotcss.drop_duplicates()
  plotcsscopy = plotcss.copy()

  # Set 'Kompetensi' as index for proper bar plotting
  plotcss_idx = plotcsscopy.set_index('LO')
# Make sure plottop3 is sorted
  plottop3 = plottop3.sort_values(by='Frekuensi', ascending=False)
  plotcss_idx = plotcss_idx.sort_values(by='Cosine Similarity', ascending=False)

  # Create an Altair bar chart
  chart2 = alt.Chart(plottop3.reset_index()).mark_bar().encode(
    x=alt.X('Frekuensi:Q', title="Frekuensi"),
    y=alt.Y('Kompetensi:N', sort='-x', title="Kompetensi")
    ).properties(width=1000, height=800)

  st.altair_chart(chart2, use_container_width=True)
  # Plot Dashboard
  # st.bar_chart(plottop3,x_label="Frekuensi",y_label="Kompetensi",
  #              horizontal=True,use_container_width=True)
  st.dataframe(data=plottop3)

  chart3 = alt.Chart(plotcss_idx.reset_index()).mark_bar().encode(
    x=alt.X('Cosine Similarity:Q', title="Cosine Similarity"),
    y=alt.Y('LO:N', sort='-x', title="LO")
    ).properties(width=1000, height=800)
  
  st.write(
    """
    # Visualisasi Cosine Similarities
    Cosine Similarity dihitung untuk setiap LO dengan meninjau relevansi LO dan 23 kompetensi yang ada. Adapun data yang digunakan adalah data survei alumni angkatan 2017.
    """
  )

  st.altair_chart(chart3, use_container_width=True)
  
  # st.bar_chart(plotcss_idx,x_label="Cosine Similarity",y_label="LO",
  #              horizontal=True,use_container_width=True)
  st.dataframe(data=plotcss)


