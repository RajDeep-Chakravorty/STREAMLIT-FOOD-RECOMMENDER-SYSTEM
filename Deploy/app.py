import streamlit as st
import pandas as pd 
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


import streamlit as st

# Define the URL for the images
left_sidebar_image_url = "https://img.freepik.com/free-photo/lavash-doner-topped-with-melted-cheese-tomato_140725-44210.jpg?t=st=1713504172~exp=1713507772~hmac=5766e213c38efeac59a7034e0ed7a52aafb110a9f606e1df94f265c8d06f5f10&w=360"
right_sidebar_image_url = "https://img.freepik.com/free-photo/top-view-eid-al-fitr-celebration-with-delicious-food_23-2151205140.jpg?t=st=1713504885~exp=1713508485~hmac=8a34e9ddb564a29ec226b36cb52ed4d5f8837543a5e6de79235d405e4ce6ff8a&w=360"


# Define the content for the left sidebar
left_sidebar_content = f"""
<div style="position: fixed; left: 0; top: 0; height: 100%; width: 20%; background-color: #FFDAB9; padding: 0px;">
    <img src="{left_sidebar_image_url}" alt="Left Sidebar Image" width="310" height="700" style="margin-left: 0;">
</div>
"""
st.markdown(left_sidebar_content, unsafe_allow_html=True)


# Define the content for the right sidebar
right_sidebar_content = f"""
<div style="position: fixed; right: 0; top: 0; height: 100%; width: 20%; background-color: #ADD8E6; padding: 20px;">
    <img src="{right_sidebar_image_url}" alt="Right Sidebar Image" width="310" height="700" style="margin-right: 0;">
</div>
"""
st.markdown(right_sidebar_content, unsafe_allow_html=True)



header_image = "https://github.com/RajDeep-Chakravorty/STREAMLIT-FOOD-RECOMMENDER-SYSTEM/raw/main/Header.png"
st.markdown(f'<div style="display: flex; justify-content: center;"><img src="{header_image}" style="width: 200px; height: auto;"></div>', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: red;'>Food Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: green;'>Let us help you with ordering</p>", unsafe_allow_html=True)
st.markdown("<center><img src='https://github.com/RajDeep-Chakravorty/STREAMLIT-FOOD-RECOMMENDER-SYSTEM/raw/main/Deploy/foood.jpg' width=700>", unsafe_allow_html=True)

st.subheader("Whats your preference?")
vegn = st.radio("Vegetables or none!", ["veg", "non-veg"], index=1) 

st.subheader("What Cuisine do you prefer?")
cuisine = st.selectbox("Choose your favourite!", ['Healthy Food', 'Snack', 'Dessert', 'Japanese', 'Indian', 'French',
       'Mexican', 'Italian', 'Chinese', 'Beverage', 'Thai'])

st.subheader("How well do you want the dish to be?")  #RATING
val = st.slider("from poor to the best!", 0, 10)

food_url = "https://raw.githubusercontent.com/RajDeep-Chakravorty/STREAMLIT-FOOD-RECOMMENDER-SYSTEM/main/Input/food.csv"
ratings_url = "https://raw.githubusercontent.com/RajDeep-Chakravorty/STREAMLIT-FOOD-RECOMMENDER-SYSTEM/main/Input/ratings.csv"

food = pd.read_csv(food_url)
ratings = pd.read_csv(ratings_url)
combined = pd.merge(ratings, food, on='Food_ID')

ans = combined.loc[(combined.C_Type == cuisine) & (combined.Veg_Non == vegn) & (combined.Rating >= val),['Name','C_Type','Veg_Non']]
names = ans['Name'].tolist()
x = np.array(names)
ans1 = np.unique(x)

finallist = ""
bruh = st.checkbox("Choose your Dish")
if bruh == True:
    finallist = st.selectbox("Our Choices", ans1)

dataset = ratings.pivot_table(index='Food_ID', columns='User_ID', values='Rating')
dataset.fillna(0, inplace=True)
csr_dataset = csr_matrix(dataset.values)
dataset.reset_index(inplace=True)

model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model.fit(csr_dataset)

def food_recommendation(Food_Name):
    n = 10
    FoodList = food[food['Name'].str.contains(Food_Name)]  
    if len(FoodList):        
        Foodi= FoodList.iloc[0]['Food_ID']
        Foodi = dataset[dataset['Food_ID'] == Foodi].index[0]
        distances , indices = model.kneighbors(csr_dataset[Foodi], n_neighbors=n+1)    
        Food_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        Recommendations = []
        for val in Food_indices:
            Foodi = dataset.iloc[val[0]]['Food_ID']
            i = food[food['Food_ID'] == Foodi].index
            Recommendations.append({'Name': food.iloc[i]['Name'].values[0], 'Distance': val[1]})
        df = pd.DataFrame(Recommendations, index=range(1, n+1))
        return df['Name']
    else:
        return "No Similar Foods."

if bruh == True:
    bruh1 = st.checkbox("We also Recommend : ")
    if bruh1 == True:
        display = food_recommendation(finallist)
        for i in display:
            st.write(i)
