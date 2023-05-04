import numpy as np
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu
import json
from sklearn.metrics import r2_score
from streamlit_lottie import st_lottie

with open( "C:/Users/rusha/Desktop/Diet recommendation/style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
load_file=load_lottiefile("C:/Users/rusha/Desktop/Diet recommendation/diet_animation.json")
st_lottie(
  load_file,
  speed=1,
  quality="high",
  height=250,
  width=500
)
#creating option menu
with st.sidebar:
  selected=option_menu('Diet recommendation system',[
      'Healthy diet',
      'Veg Healthy diet',
      'non-veg healthy diet',
      'Weight Gain diet',
      'Veg Weight Gain diet',
      'Non-Veg Weight Gain diet',
      'Weight loss diet',
      'Veg Weight loss diet',
      'Non-Veg Weight loss diet'
  ],default_index=0)





gender=st.radio(
  "Select gender:",
  ('Male','Female')
)







# loaded_model=pickle.load(open('C:/Users/rusha/Desktop/Test streamlit/test.sav','rb'))

df1=pd.read_excel("food_items.xlsx")
df1.drop(0,axis='rows',inplace=True)
df1.sort_values("VegNovVeg",inplace=True,ignore_index=True)
onlyveg=df1.iloc[:70]

healthyonlyveg=df1.iloc[:71]


km=KMeans(n_clusters=4,random_state=42)
healthy_pred=km.fit_predict(healthyonlyveg[['Proteins','Calories']])
healthyonlyveg['Clusters']=healthy_pred
xheal=healthyonlyveg['Clusters']
xheal.reset_index(drop=True,inplace=True)

onlynonveg=df1.iloc[70:]
onlynonveg.reset_index(drop=True, inplace=True)

km=KMeans(n_clusters=4,random_state=42)
y_pred=km.fit_predict(onlyveg[['Proteins','Calories']])

#adding predictions to dataframe
onlyveg['Clusters']=y_pred


#for nonveg only
km=KMeans(n_clusters=4,random_state=42)
y_prednonveg=km.fit_predict(onlynonveg[['Proteins','Calories']])

#adding predictions to dataframe
onlynonveg['Clusters']=y_prednonveg

#for both
km=KMeans(n_clusters=4,random_state=50)
y_predboth=km.fit_predict(df1[['Proteins','Calories']])



#adding predictions to dataframe
df1['Clusters']=y_predboth

x=onlyveg['Clusters']
x.reset_index(drop=True,inplace=True)

bothx=df1['Clusters']
bothx.reset_index(drop=True,inplace=True)

xnonv=onlynonveg['Clusters']
xnonv.reset_index(drop=True,inplace=True)


#healthy Diet
def healthydiet(bmi,bmr):
  #Creating x and y variable for input and output respectivily
  X=df1.drop(['Food_items','Clusters','Breakfast','Lunch','Dinner','VegNovVeg'],axis='columns')
  y=df1['Clusters']
  
  
  #training and testing 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

  #intializing randomforest
  model=RandomForestClassifier(n_estimators=100,criterion= "gini")
  model.fit(X_train,y_train)

  y_pred=model.predict(X_test)
  # # return r2_score(y_test, y_pred)
  # return onlyveg
  # return df1
  if bmi<=18.5:
    lst = pd.DataFrame()
    for i in range(len(bothx)):
      if(bothx[i]==3):
        lst=lst.append({"Name":df1['Food_items'][i],"Calories":df1['Calories'][i],"Proteins":df1['Proteins'][i]},ignore_index=True)
  
    temp=pd.DataFrame(columns=['Namee','Caloriess','Proteinss'])
    count=0
    for i in range(len(lst)):
      while(bmr>count):
        temp=temp.append({'Namee':lst['Name'][i],'Caloriess':lst['Calories'][i],"Proteinss":lst['Proteins'][i]},ignore_index=True)
        count=temp['Caloriess'].sum()
        i=i+1
    return temp
  
  elif(bmi>18.5 and bmi<=24.9):
    lst1 = pd.DataFrame()
    for i in range(len(bothx)):
      if(bothx[i]==0):
        lst1=lst1.append({"Name":df1['Food_items'][i],"Calories":df1['Calories'][i],"Proteins":df1['Proteins'][i]},ignore_index=True)
  
    # print(lst)
  
    temp=pd.DataFrame(columns=['Namee','Caloriess','Proteinss'])
    count=0
    for i in range(len(lst1)):
      while(bmr>count):
        temp=temp.append({'Namee':lst1['Name'][i],'Caloriess':lst1['Calories'][i],"Proteinss":lst1['Proteins'][i]},ignore_index=True)
        count=temp['Caloriess'].sum()
        i=i+1
    return temp

  elif(bmi>=25 and bmi<=38):
    lst = pd.DataFrame()
    for i in range(len(bothx)):
      if(bothx[i]==1):
        lst=lst.append({"Name":df1['Food_items'][i],"Calories":df1['Calories'][i],"Proteins":df1['Proteins'][i]},ignore_index=True)
    # return lst
    temp=pd.DataFrame(columns=['Name','Calories','Proteins'])
    count=0

    for i in range(len(lst)):
      while(bmr>count):
        temp=temp.append({'Name':lst['Name'][i],'Calories':lst['Calories'][i],'Proteins':lst['Proteins'][i]},ignore_index=True)
        count=temp['Calories'].sum()
        i=i+1
    return temp

def veghealthydiet(bmi,bmr):
  # km=KMeans(n_clusters=4)
  # y_pred=km.fit_predict(onlyveg[['Proteins','Calories']])
  # onlyveg['Clusters']=y_pred
  # x=onlyveg['Clusters']
  
  #Creating x and y variable for input and output respectivily
  X=onlyveg.drop(['Food_items','Clusters','Breakfast','Lunch','Dinner','VegNovVeg'],axis='columns')
  y=onlyveg['Clusters']

  #training and testing 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

  #intializing randomforest
  model=RandomForestClassifier(n_estimators=100,criterion= "gini")
  model.fit(X_train,y_train)

  

  # return x
  if bmi<=18.5:
    lst = pd.DataFrame()
    for i in range(len(xheal)):
      if(xheal[i]==3):
        lst=lst.append({"Name":healthyonlyveg['Food_items'][i],"Calories":healthyonlyveg['Calories'][i],"Proteins":healthyonlyveg['Proteins'][i],"Fats":healthyonlyveg['Fats'][i]},ignore_index=True)
  
    # return lst
    temp=pd.DataFrame(columns=['Namee','Caloriess','Fats'])
    count=0

    for i in range(len(lst)):
      while(bmr>count):
        temp=temp.append({'Namee':lst['Name'][i],'Caloriess':lst['Calories'][i],'Fats':lst['Fats'][i]},ignore_index=True)
        count=temp['Caloriess'].sum()
        i=i+1
    return temp

  elif(bmi>18.5 and bmi<=24.9):
    lst1 = pd.DataFrame()
    for i in range(len(xheal)):
      if(xheal[i]==2):
        lst1=lst1.append({"Name":healthyonlyveg['Food_items'][i],"Calories":healthyonlyveg['Calories'][i],"Proteins":healthyonlyveg['Proteins'][i]},ignore_index=True)
  
    return lst1
  
    # temp=pd.DataFrame(columns=['Name','Calories','Proteins'])
    # count=0
    # for i in range(len(lst1)):
    #   while(bmr>count):
    #     temp=temp.append({'Name':lst1['Name'][i],'Calories':lst1['Calories'][i],"Proteins":lst1['Proteins'][i]},ignore_index=True)
    #     count=temp['Calories'].sum()
    #     i=i+1
    # return temp

  elif(bmi>=25 and bmi<=38):
    lst = pd.DataFrame()
    for i in range(len(xheal)):
      if(xheal[i]==0):
        lst=lst.append({"Name":healthyonlyveg['Food_items'][i],"Calories":healthyonlyveg['Calories'][i],"Fats":healthyonlyveg['Fats'][i]},ignore_index=True)
  
    temp=pd.DataFrame(columns=['Namee','Caloriess','Fats'])
    count=0

    for i in range(len(lst)):
      while(bmr>count):
        temp=temp.append({'Namee':lst['Name'][i],'Caloriess':lst['Calories'][i],'Fats':lst['Fats'][i]},ignore_index=True)
        count=temp['Caloriess'].sum()
        i=i+1
    return temp
   
def nonveghealthydiet(bmi):
  #Creating x and y variable for input and output respectivily
  X=onlynonveg.drop(['Food_items','Clusters','Breakfast','Lunch','Dinner','VegNovVeg'],axis='columns')
  y=onlynonveg['Clusters']

  #training and testing 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

  #intializing randomforest
  model=RandomForestClassifier(n_estimators=100,criterion= "gini")
  model.fit(X_train,y_train)


  #recommend based on bmi
  if bmi<=18.5:
    lst = pd.DataFrame()
    for i in range(len(xnonv)):
      if(xnonv[i]==1 or xnonv[i]==2):
        lst=lst.append({"Name":onlynonveg['Food_items'][i],"Calories":onlynonveg['Calories'][i],"Proteins":onlynonveg['Proteins'][i],"Fats":onlynonveg['Fats'][i]},ignore_index=True)
    return lst

  elif(bmi>18.5 and bmi<=24.9):
      lst1 = pd.DataFrame()
      for i in range(len(xnonv)):
        if(xnonv[i]==0):
          lst1=lst1.append({"Name":onlynonveg['Food_items'][i],"Calories":onlynonveg['Calories'][i],"Proteins":onlynonveg['Proteins'][i]},ignore_index=True)
  
    # print(lst)
      lst1.reset_index(drop=True,inplace=True)
      return lst1
      


  elif(bmi>=25 and bmi<=38):
      lst = pd.DataFrame()
      for j in range(len(xnonv)):
        if(xnonv[j]==3):
          lst=lst.append({"Name":onlynonveg['Food_items'][j],"Calories":onlynonveg['Calories'][j],"Proteins":onlynonveg['Proteins'][j]},ignore_index=True)
  
      # temp=pd.DataFrame(columns=['Namee','Caloriess','Fats'])
      return lst

#***************************************************************************************************************#
#***************************************************************************************************************#


#weight gain
#reading csv
wgdf=pd.read_excel("food_items.xlsx")

#droping 1st records
wgdf.drop(0,axis='rows',inplace=True)

#reseting index
wgdf.reset_index(drop=True,inplace=True)

#performing clustering
km=KMeans(n_clusters=4,random_state=42)
y_pred=km.fit_predict(wgdf[['Proteins','Calories','Carbohydrates','Fats']])

#adding new column to df
wgdf['Clusters']=y_pred
x=wgdf['Clusters']

#for only veg
wgonlyveg=onlyveg.drop(['Clusters'],axis=1)
km=KMeans(n_clusters=4,random_state=42)
y_predveg=km.fit_predict(wgonlyveg[['Proteins','Calories','Carbohydrates','Fats']])
wgonlyveg['Clusters']=y_predveg
xveg=wgonlyveg['Clusters']
xveg.reset_index(drop=True,inplace=True)

#for only non veg
wgonlynonveg=onlynonveg
km=KMeans(n_clusters=4,random_state=42)
y_prednonveg=km.fit_predict(wgonlynonveg[['Proteins','Calories','Carbohydrates','Fats']])
wgonlynonveg['Clusters']=y_prednonveg
xwgnon=wgonlynonveg['Clusters']
xwgnon.reset_index(drop=True,inplace=True)

def weightgainboth(bmi):
  #craeting ip 
  X=wgdf.drop(['Food_items','Clusters','Breakfast','Lunch','Dinner','VegNovVeg'],axis='columns')

  #creating op
  y=wgdf['Clusters']

  #training and testing
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

  #intialize model
  model=RandomForestClassifier(n_estimators=100,criterion= "gini")
  model.fit(X_train,y_train)

  #creating new food dataframe
  food=wgdf[['Food_items','Proteins','Calories','Carbohydrates']]
  

  if(bmi<=18.5):
    lst = pd.DataFrame()
    for i in range(len(x)):
      if(x[i]==1):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Proteins":food['Proteins'][i],"Calories":food['Calories'][i],"Carbohydrates":food['Carbohydrates'][i]},ignore_index=True)
    return lst

  elif(bmi>18.5 and bmi<=24.9):
    lst = pd.DataFrame()
    for i in range(len(x)):
      if(x[i]==2 or x[i]==3):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Proteins":food['Proteins'][i],"Calories":food['Calories'][i],"Carbohydrates":food['Carbohydrates'][i]},ignore_index=True)
    return lst

  elif(bmi>=25 and bmi<=38):
    lst = pd.DataFrame()
    print("We suggest you to first lose some weight and then try to gain muscles")
    for i in range(len(x)):
      if(x[i]==3 or x[i]==0):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Proteins":food['Proteins'][i],"Calories":food['Calories'][i],"Carbohydrates":food['Carbohydrates'][i]},ignore_index=True)
    return lst

def weightgainveg(bmi):
  X=wgonlyveg.drop(['Food_items','Breakfast','Lunch','Dinner','VegNovVeg'],axis='columns')

  #creating op
  y=wgonlyveg['Clusters']

  #training and testing
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

  #intialize model
  model=RandomForestClassifier(n_estimators=100,criterion= "gini")
  model.fit(X_train,y_train)

  #creating new food dataframe
  food=wgonlyveg[['Food_items','Proteins','Calories','Carbohydrates']]
  
  
  if(bmi<=18.5):
    lst = pd.DataFrame()
    for i in range(len(xveg)):
      if(xveg[i]==3):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Proteins":food['Proteins'][i],"Calories":food['Calories'][i],"Carbohydrates":food['Carbohydrates'][i]},ignore_index=True)
    return lst

  elif(bmi>18.5 and bmi<=24.9):
    lst = pd.DataFrame()
    for i in range(len(xveg)):
      if(xveg[i]==0):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Proteins":food['Proteins'][i],"Calories":food['Calories'][i],"Carbohydrates":food['Carbohydrates'][i]},ignore_index=True)
    return lst

  elif(bmi>=25 and bmi<=38):
    lst = pd.DataFrame()
    print("We suggest you to first lose some weight and then try to gain muscles")
    for i in range(len(xveg)):
      if(xveg[i]==1):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Proteins":food['Proteins'][i],"Calories":food['Calories'][i],"Carbohydrates":food['Carbohydrates'][i]},ignore_index=True)
    return lst

def weightgainnonveg(bmi):
  X=wgonlynonveg.drop(['Food_items','Breakfast','Lunch','Dinner','VegNovVeg'],axis='columns')

  #creating op
  y=wgonlynonveg['Clusters']

  #training and testing
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

  #intialize model
  model=RandomForestClassifier(n_estimators=100,criterion= "gini")
  model.fit(X_train,y_train)

  #creating new food dataframe
  food=wgonlynonveg[['Food_items','Proteins','Calories','Carbohydrates']]
  
  
  if(bmi<=18.5):
    lst = pd.DataFrame()
    for i in range(len(xwgnon)):
      if(xwgnon[i]==2 or xwgnon[i]==0):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Proteins":food['Proteins'][i],"Calories":food['Calories'][i],"Carbohydrates":food['Carbohydrates'][i]},ignore_index=True)
    return food

  elif(bmi>18.5 and bmi<=24.9):
    lst = pd.DataFrame()
    for i in range(len(xwgnon)):
      if(xwgnon[i]==0 or xwgnon[i]==1):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Proteins":food['Proteins'][i],"Calories":food['Calories'][i],"Carbohydrates":food['Carbohydrates'][i]},ignore_index=True)
    return lst

  elif(bmi>=25 and bmi<=38):
    lst = pd.DataFrame()
    print("We suggest you to first lose some weight and then try to gain muscles")
    for i in range(len(xwgnon)):
      if(xwgnon[i]==3):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Proteins":food['Proteins'][i],"Calories":food['Calories'][i],"Carbohydrates":food['Carbohydrates'][i]},ignore_index=True)
    return lst
    
#***************************************************************************************************************#
#***************************************************************************************************************#


#weight loss
#reading csv
wldf=pd.read_excel("food_items.xlsx")

  #droping 1st records
wldf.drop(0,axis='rows',inplace=True)

wldf.sort_values("VegNovVeg",inplace=True,ignore_index=True)

wlonlyveg=wldf.iloc[:70]

wlonlynonveg=wldf.iloc[70:]

  #reseting index
wldf.reset_index(drop=True,inplace=True)

  #performing clustering
km=KMeans(n_clusters=4,random_state=42)
y_pred=km.fit_predict(wldf[['Calories','Sugars','Fats']])

 #adding new column to df
wldf['Clusters']=y_pred
wlx=wldf['Clusters']
wlx.reset_index(drop=True,inplace=True)

km=KMeans(n_clusters=4,random_state=42)
y_predveg=km.fit_predict(wlonlyveg[['Calories','Sugars','Fats']])
wlonlyveg['Clusters']=y_predveg
wlveg=wlonlyveg['Clusters']
wlveg.reset_index(drop=True,inplace=True)

wlonlynonveg.reset_index(drop=True,inplace=True)
km=KMeans(n_clusters=4,random_state=42)
y_prednonveg=km.fit_predict(wlonlynonveg[['Calories','Sugars','Fats']])
wlonlynonveg['Clusters']=y_prednonveg
wlnveg=wlonlynonveg['Clusters']

def weightlossboth(bmi):
  #craeting ip 
  X=wldf.drop(['Food_items','Clusters','Breakfast','Lunch','Dinner','VegNovVeg'],axis='columns')

  #creating op
  y=wldf['Clusters']

  #training and testing
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

  #intialize model
  model=RandomForestClassifier(n_estimators=100,criterion= "gini")
  model.fit(X_train,y_train)

  #creating new food dataframe
  food=wldf[['Food_items','Proteins','Calories','Carbohydrates','Fats','Sugars']]
  

  #recommendation
  if(bmi<=18.5):
    lst = pd.DataFrame()
    print("We suggest you first to gain or try to increase your body mass index before lossing fats")
    for i in range(len(wlx)):
      if(wlx[i]==0):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Proteins":food['Proteins'][i],"Calories":food['Calories'][i],"Fats":food['Fats'][i],"Sugars":food['Sugars'][i]},ignore_index=True)
    return lst

  elif(bmi>18.5 and bmi<=24.9):
    lst = pd.DataFrame()
    for i in range(len(wlx)):
      if(wlx[i]==2 or wlx[i]==3):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Proteins":food['Proteins'][i],"Calories":food['Calories'][i],"Carbohydrates":food['Carbohydrates'][i]},ignore_index=True)
    return lst

  elif(bmi>=25 and bmi<=38):
    lst = pd.DataFrame()
    print("We suggest you to first lose some weight and then try to gain muscles")
    for i in range(len(wlx)):
      if(wlx[i]==2):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Proteins":food['Proteins'][i],"Calories":food['Calories'][i],"Carbohydrates":food['Carbohydrates'][i]},ignore_index=True)
    return lst

def weightlossveg(bmi):
  #craeting ip 
  X=wlonlyveg.drop(['Food_items','Clusters','Breakfast','Lunch','Dinner','VegNovVeg'],axis='columns')

  #creating op
  y=wlonlyveg['Clusters']

  #training and testing
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

  #intialize model
  model=RandomForestClassifier(n_estimators=100,criterion= "gini")
  model.fit(X_train,y_train)

  #creating new food dataframe
  food=wlonlyveg[['Food_items','Proteins','Calories','Carbohydrates','Fats','Sugars']]
  # x=wlonlyveg['Clusters']

  #recommendation
  if(bmi<=18.5):
    lst = pd.DataFrame()
    print("We suggest you first to gain or try to increase your body mass index before lossing fats")
    for i in range(len(wlveg)):
      if(wlveg[i]==2):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Calories":food['Calories'][i],"Fats":food['Fats'][i]},ignore_index=True)
    return lst

  elif(bmi>18.5 and bmi<=24.9):
    lst = pd.DataFrame()
    for i in range(len(wlveg)):
      if(wlveg[i]==1 or wlveg[i]==2):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Calories":food['Calories'][i],"Fats":food['Fats'][i]},ignore_index=True)
    return lst

  elif(bmi>=25 and bmi<=38):
    lst = pd.DataFrame()
    for i in range(len(wlveg)):
      if(wlveg[i]==1):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Calories":food['Calories'][i],"Fats":food['Fats'][i]},ignore_index=True)
    return lst

def weightlossnonveg(bmi):
  #craeting ip 
  X=wlonlynonveg.drop(['Food_items','Clusters','Breakfast','Lunch','Dinner','VegNovVeg'],axis='columns')

  #creating op
  y=wlonlynonveg['Clusters']

  #training and testing
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

  #intialize model
  model=RandomForestClassifier(n_estimators=100,criterion= "gini")
  model.fit(X_train,y_train)

  y_predtt=model.predict(X_test)
  #creating new food dataframe
  food=wlonlynonveg[['Food_items','Proteins','Calories','Carbohydrates','Fats','Sugars']]
  
  # r2=r2_score(y_test, y_predtt)
  # return r2 
  #recommendation
  if(bmi<=18.5):
    lst = pd.DataFrame()
    print("We suggest you first to gain or try to increase your body mass index before lossing fats")
    for i in range(len(wlnveg)):
      if(wlnveg[i]==0):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Calories":food['Calories'][i],"Fats":food['Fats'][i]},ignore_index=True)
    return lst

  elif(bmi>18.5 and bmi<=24.9):
    lst = pd.DataFrame()
    for i in range(len(wlnveg)):
      if(wlnveg[i]==1):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Calories":food['Calories'][i],"Fats":food['Fats'][i]},ignore_index=True)
    return lst

  elif(bmi>=25 and bmi<=38):
    lst = pd.DataFrame()
    for i in range(len(wlnveg)):
      if(wlnveg[i]==3):
        # lst.append(food[i])
        lst=lst.append({"Name":food['Food_items'][i],"Calories":food['Calories'][i],"Fats":food['Fats'][i]},ignore_index=True)
    return lst


#Diet selection part

if selected=='Healthy diet':
  st.title("Healthy diet")
  height=st.number_input('Enter your height')
  weight=st.number_input('Enter your weight')
  age=st.number_input('Enter your age')
  bmii=0
  if(height>0 or weight>0):
    bmii=weight/((height/100)**2) 


  if(gender=='Male'):
    bmr = (10 * weight) + (6.25 * height ) - (5 * age) + 5
  else:
    bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
  
  st.write("Body mass index: ",bmii)
  st.write("Basal Metabolic Rate",bmr)
  # bmiip=st.number_input('Enter your bmi')
  test = pd.DataFrame()
  if st.button('recommend'):
    test=healthydiet(int(bmii),int(bmr))
  
  # st.success(test)
  st.markdown(hide_table_row_index, unsafe_allow_html=True)
  st.table(test)


if selected=='Veg Healthy diet':
  st.title("Veg Healthy diet")
  height=st.number_input('Enter your height')
  weight=st.number_input('Enter your weight')
  age=st.number_input('Enter your age')
  bmii=0
  if(height>0 or weight>0):
    bmii=weight/((height/100)**2) 
  

  if(gender=='Male'):
    bmr = (10 * weight) + (6.25 * height ) - (5 * age) + 5
  else:
    bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
  
  st.write("Body mass index: ",bmii)
  st.write("Basal Metabolic Rate",bmr)
  # bmiip=st.number_input('Enter your bmi')
  test = pd.DataFrame()
  if st.button('recommend'):
    test=veghealthydiet(int(bmii),int(bmr))
  
  # st.success(test)
  st.markdown(hide_table_row_index, unsafe_allow_html=True)
  st.table(test)

if selected=='non-veg healthy diet':
  st.title("non-veg healthy diet")
  height=st.number_input('Enter your height')
  weight=st.number_input('Enter your weight')
  age=st.number_input('Enter your age')
  bmii=0
  if(height>0 or weight>0):
    bmii=weight/((height/100)**2) 


  if(gender=='Male'):
    bmr = (10 * weight) + (6.25 * height ) - (5 * age) + 5
  else:
    bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
  
  st.write("Body mass index: ",bmii)
  st.write("Basal Metabolic Rate",bmr)
  # bmiip=st.number_input('Enter your bmi')
  test = pd.DataFrame()
  if st.button('recommend'):
    test=nonveghealthydiet(int(bmii))
  
  # st.success(test)
  st.markdown(hide_table_row_index, unsafe_allow_html=True)
  st.table(test)

if selected=='Weight Gain diet':
  st.title("Weight Gain diet")
  height=st.number_input('Enter your height')
  weight=st.number_input('Enter your weight')
  age=st.number_input('Enter your age')
  bmii=0
  if(height>0 or weight>0):
    bmii=weight/((height/100)**2) 


  if(gender=='Male'):
    bmr = (10 * weight) + (6.25 * height ) - (5 * age) + 5
  else:
    bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
  
  st.write("Body mass index: ",bmii)
  # st.write("Basal Metabolic Rate",bmr)
  # bmiip=st.number_input('Enter your bmi')
  test = pd.DataFrame()
  if st.button('recommend'):
    test=weightgainboth(int(bmii))
  
  # st.success(test)
  st.markdown(hide_table_row_index, unsafe_allow_html=True)
  st.table(test)

if selected=='Veg Weight Gain diet':
  st.title("Veg Weight Gain diet")
  height=st.number_input('Enter your height')
  weight=st.number_input('Enter your weight')
  age=st.number_input('Enter your age')
  bmii=0
  if(height>0 or weight>0):
    bmii=weight/((height/100)**2) 


  if(gender=='Male'):
    bmr = (10 * weight) + (6.25 * height ) - (5 * age) + 5
  else:
    bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
  
  st.write("Body mass index: ",bmii)
  # st.write("Basal Metabolic Rate",bmr)
  # bmiip=st.number_input('Enter your bmi')
  test = pd.DataFrame()
  if st.button('recommend'):
    test=weightgainveg(int(bmii))
  
  # st.success(test)
  st.markdown(hide_table_row_index, unsafe_allow_html=True)
  st.table(test)
  
if selected=='Non-Veg Weight Gain diet':
  st.title("Non-Veg Weight Gain diet")
  height=st.number_input('Enter your height')
  weight=st.number_input('Enter your weight')
  age=st.number_input('Enter your age')
  bmii=0
  if(height>0 or weight>0):
    bmii=weight/((height/100)**2) 


  if(gender=='Male'):
    bmr = (10 * weight) + (6.25 * height ) - (5 * age) + 5
  else:
    bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
  
  st.write("Body mass index: ",bmii)
  # st.write("Basal Metabolic Rate",bmr)
  # bmiip=st.number_input('Enter your bmi')
  test = pd.DataFrame()
  if st.button('recommend'):
    test=weightgainnonveg(int(bmii))
  
  # st.success(test)
  st.markdown(hide_table_row_index, unsafe_allow_html=True)
  st.table(test)

if selected=='Weight loss diet':
  st.title("Weight loss diet")
  height=st.number_input('Enter your height')
  weight=st.number_input('Enter your weight')
  age=st.number_input('Enter your age')
  bmii=0
  if(height>0 or weight>0):
    bmii=weight/((height/100)**2) 


  if(gender=='Male'):
    bmr = (10 * weight) + (6.25 * height ) - (5 * age) + 5
  else:
    bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
  
  st.write("Body mass index: ",bmii)
  # st.write("Basal Metabolic Rate",bmr)
  # bmiip=st.number_input('Enter your bmi')
  test = pd.DataFrame()
  if st.button('recommend'):
    test=weightlossboth(int(bmii))
  
  # st.success(test)
  st.markdown(hide_table_row_index, unsafe_allow_html=True)
  st.table(test)

if selected=='Veg Weight loss diet':
  st.title("Veg Weight loss diet")
  height=st.number_input('Enter your height')
  weight=st.number_input('Enter your weight')
  age=st.number_input('Enter your age')
  bmii=0
  if(height>0 or weight>0):
    bmii=weight/((height/100)**2) 


  if(gender=='Male'):
    bmr = (10 * weight) + (6.25 * height ) - (5 * age) + 5
  else:
    bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
  
  st.write("Body mass index: ",bmii)
  # st.write("Basal Metabolic Rate",bmr)
  # bmiip=st.number_input('Enter your bmi')
  test = pd.DataFrame()
  if st.button('recommend'):
    test=weightlossveg(int(bmii))
  
  # st.success(test)
  st.markdown(hide_table_row_index, unsafe_allow_html=True)
  st.table(test)

if selected=='Non-Veg Weight loss diet':
  st.title("Non-Veg Weight loss diet")
  height=st.number_input('Enter your height')
  weight=st.number_input('Enter your weight')
  age=st.number_input('Enter your age')
  bmii=0
  if(height>0 or weight>0):
    bmii=weight/((height/100)**2) 


  if(gender=='Male'):
    bmr = (10 * weight) + (6.25 * height ) - (5 * age) + 5
  else:
    bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
  
  st.write("Body mass index: ",bmii)
  # st.write("Basal Metabolic Rate",bmr)
  # bmiip=st.number_input('Enter your bmi')
  test = pd.DataFrame()
  if st.button('recommend'):
    test=weightlossnonveg(int(bmii))
  
  # st.success(test)
  st.markdown(hide_table_row_index, unsafe_allow_html=True)
  st.table(test)
#need to incrase range of last bmi which is 25 to 29 to most proably 25 to 40














# def main():
#   # st.title("First Try")
#   # height=st.number_input('Enter your height')
#   # weight=st.number_input('Enter your weight')
#   # age=st.number_input('Enter your age')
#   # bmii=weight/((height/100)**2) 
#   # if(gender=='Male'):
#   #   bmr = (10 * weight) + (6.25 * height ) - (5 * age) + 5
#   # else:
#   #   bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
  
#   # st.write("Body mass index: ",bmii)
#   # st.write("Basal Metabolic Rate",bmr)
#   # # bmiip=st.number_input('Enter your bmi')
#   # test = pd.DataFrame()
#   # if st.button('recommend'):
#   #   test=veghealthydiet(int(bmii),int(bmr))
  
#   # # st.success(test)
#   # st.table(test)

# if __name__=='__main__':
#   main()
