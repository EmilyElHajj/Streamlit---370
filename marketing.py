import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import datetime
from datetime import date
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


import warnings
warnings.filterwarnings("ignore")
np.random.seed(1)

st.set_page_config(layout='wide')
st.markdown(
   """
   <style>
   .main {
   background-color:#d0cfe3;
   }
   </style>
   """
   ,
   unsafe_allow_html=True
)

st.title("Marketing Analytics Dashboard")
st.write("Via this dashboard, you as a marketer will gain better insights on your customers **demographics**, **behaviour** and their **preferred marketing campaigns**. You will also get the chance to predict  based on the customer demographics if they will respond to a campaign or not. ")
st.write("Explore the dataset content and columns.")

df = pd.read_csv(r"marketing_campaign - Copy.csv")

df.dropna(subset = ["Income"], inplace=True)
df=df[(df['Income']<600000) & (df['Income']  >3000)]

conditions = [
    (df['Year_Birth'] >= 1893) & (df['Year_Birth'] <= 1945),
    (df['Year_Birth'] >= 1946) & (df['Year_Birth'] <= 1964),
    (df['Year_Birth'] >= 1965) & (df['Year_Birth'] <= 1979),
    (df['Year_Birth'] >= 1980) & (df['Year_Birth'] <= 1995),
    (df['Year_Birth'] >= 1996) & (df['Year_Birth'] <= 2009)
    ]

# create a list of the values we want to assign for each condition
values = ['Silent', 'Baby Bommer', 'GenZ','GenY','GenZ']

# create a new column and use np.select to assign values to it
df['Generation'] = np.select(conditions, values)


conditions = [
    (df['Income'] >= 1000) & (df['Income'] <= 30000),
    (df['Income'] > 30000) & (df['Income'] <= 60000),
    (df['Income'] > 60000) & (df['Income'] <= 100000),
    (df['Income'] > 100000) & (df['Income'] <= 600000)
 ]

# create a list of the values we want to assign for each condition
values = ['Below average', 'Average earners', 'Middle class','High earners']

# create a new column and use np.select to assign values to it
df['Income_categories'] = np.select(conditions, values)
#df.drop(df['Income_categories'] == '0')

#2 new columns, spending and transactions
df['Purchases']=df['MntWines']+df['MntFruits']+df['MntMeatProducts']+df['MntFishProducts']+df['MntSweetProducts']+df['MntGoldProds']
df['Transactions']=df['NumWebPurchases']+df['NumCatalogPurchases']+df['NumStorePurchases']

df = df[df['Transactions'] > 1] #We keep customers with repeated purchases, implying number of transactions must be at least 2
df = df[df['Purchases'] > 0]

df['AcceptedCmpTotal']=df['AcceptedCmp4']+df['AcceptedCmp5']+df['AcceptedCmp3']+df['AcceptedCmp2']+df['AcceptedCmp1']

conditions = [
    (df['Purchases'] >= 7) & (df['Purchases'] <= 200),
    (df['Purchases'] > 200) & (df['Purchases'] <= 600),
    (df['Purchases'] > 600) & (df['Purchases'] <= 1000),
    (df['Purchases'] > 1000) & (df['Purchases'] <= 2000),
    (df['Purchases'] > 2000) & (df['Purchases'] <= 2555)
 ]

# create a list of the values we want to assign for each condition
values = ['Very low purchaser','Low purchaser', 'Average purchaser', 'Good purchaser','Great Purchaser']

# create a new column and use np.select to assign values to it using our lists as arguments
df['Purchaser_categories'] = np.select(conditions, values)


if st.button('Explore Dataset'):
    st.dataframe(df.head())

###################################################################################################
with st.sidebar.header('1. Select Criteria'):
    demo = st.sidebar.selectbox('Affects fig. I III IV & V ', ['Generation','Income','Education','Marital Status','Purchaser type'])

with st.sidebar.header('2. Select Product Category'):
    demo1 = st.sidebar.selectbox('Affects fig. II only', ['Sweet','Wine','Fruits','Meat','Gold','Fish'])

demographics = st.beta_container()
purchases = st.beta_container()
platforms = st.beta_container()
ml = st.beta_container()

with demographics:
    sel_col, dis_col = st.beta_columns(2)
    sel_col.header('I. Customers Demographics')
    sel_col.subheader('Understand your customers better')
    #demo = sel_col.selectbox('Select Criteria', ['Generation','Income','Education','Marital Status'])

    if demo == 'Generation':
        fig = px.histogram(
        data_frame=df,
        x="Generation",
        y="ID",
        title="Customers Age groups",
        histfunc='count'
        )
        sel_col.write(fig)
    elif demo == 'Income':
        figb = px.histogram(
        data_frame=df,
        x="Income_categories",
        y="ID",
        title="Customers Income Categories",
        histfunc='count'
        )
        sel_col.write(figb)
    elif demo == 'Education':
        figc = px.histogram(
        data_frame=df,
        x="Education",
        y="ID",
        title="Customers Education level",
        histfunc='count'
        )
        sel_col.write(figc)
    elif demo == 'Marital Status':
        figd = px.histogram(
        data_frame=df,
        x="Marital_Status",
        y="ID",
        title="Customers Marital Status",
        histfunc='count'
        )
        sel_col.write(figd)

    elif demo == 'Purchaser type':
        figd = px.histogram(
        data_frame=df,
        x="Purchaser_categories",
        y="ID",
        title="Customers Purchaser types",
        histfunc='count'
        )
        sel_col.write(figd)


    dis_col.header('II. Customers Behaviour')
    dis_col.subheader('Product Category vs.Customer age group')
    #demo1 = dis_col.selectbox('Select Product Category', ['Sweet','Wine','Fruits','Meat','Gold','Fish'])

    if demo1 == 'Sweet':
     fig1 = px.histogram(
     data_frame=df,
     x="MntSweetProducts",
     y="MntSweetProducts",
     title="Sweet Products",
     histfunc='count',
     color='Generation'
     )
     dis_col.write(fig1)

    elif demo1 == 'Wine':
     fig2 = px.histogram(
     data_frame=df,
     x="MntWines",
     y="MntWines",
     title="Winery Products",
     histfunc='count',
     color='Generation'
     )
     dis_col.write(fig2)

    elif demo1 == 'Fruits':
     fig3 = px.histogram(
     data_frame=df,
     x="MntFruits",
     y="MntFruits",
     title="Fruit Products",
     histfunc='count',
     color='Generation'
     )
     dis_col.write(fig3)

    elif demo1 == 'Meat':
     fig4 = px.histogram(
     data_frame=df,
     x="MntMeatProducts",
     y="MntMeatProducts",
     title="Meat Products",
     histfunc='count',
     color='Generation'
     )
     dis_col.write(fig4)

    elif demo1 == 'Gold':
     fig5 = px.histogram(
     data_frame=df,
     x="MntGoldProds",
     y="MntGoldProds",
     title="Gold Products",
     histfunc='count',
     color='Generation'
     )
     dis_col.write(fig5)

    elif demo1 == 'Fish':
     fig6 = px.histogram(
     data_frame=df,
     x="MntFishProducts",
     y="MntFishProducts",
     title="Fish Products",
     histfunc='count',
     color='Generation'
     )
     dis_col.write(fig6)

with purchases:
    col1, col2 = st.beta_columns(2)
    col1.header('III. Number of Accepted marketing campaigns')
    col1.subheader('What customer demographics or behaviour type are likely to accept marketing campaigns')
    #demo2 = st.selectbox('Select Criteria', ['Income','Generation','Purchaser type','Education'])
    col2.header('IV. Number of Purchases based on Deals')
    col2.subheader('What customer demographics or behaviour type are likely to purchase based on deals offered')

    if demo == 'Income':
        fig7 = px.histogram(
        data_frame=df,
        x="AcceptedCmpTotal",
        y="AcceptedCmpTotal",
        title="Accepted marketing campaigns",
        color='Income_categories')
        col1.write(fig7)

    elif demo == 'Generation':
        fig8 = px.histogram(
        data_frame=df,
        x="AcceptedCmpTotal",
        y="AcceptedCmpTotal",
        title="Accepted marketing campaigns",
        color='Generation')
        col1.write(fig8)

    elif demo == 'Purchaser type':
        fig9 = px.histogram(
        data_frame=df,
        x="AcceptedCmpTotal",
        y="AcceptedCmpTotal",
        title="Accepted marketing campaigns",
        color='Purchaser_categories')
        col1.write(fig9)

    elif demo == 'Education':
        fig10 = px.histogram(
        data_frame=df,
        x="AcceptedCmpTotal",
        y="AcceptedCmpTotal",
        title="Accepted marketing campaigns",
        color='Education')
        col1.write(fig10)

    elif demo == 'Marital Status':
        fig11 = px.histogram(
        data_frame=df,
        x="AcceptedCmpTotal",
        y="AcceptedCmpTotal",
        title="Accepted marketing campaigns",
        color='Marital_Status')
        col1.write(fig11)

    if demo == 'Income':
        fig7 = px.histogram(
        data_frame=df,
        x="NumDealsPurchases",
        y="Purchases",
        title="Purchases based on Deals",
        color='Income_categories')
        col2.write(fig7)

    elif demo == 'Generation':
        fig8 = px.histogram(
        data_frame=df,
        x="NumDealsPurchases",
        y="Purchases",
        title="Purchases based on Deals",
        color='Generation')
        col2.write(fig8)

    elif demo == 'Purchaser type':
        fig9 = px.histogram(
        data_frame=df,
        x="NumDealsPurchases",
        y="Purchases",
        title="Purchases based on Deals",
        color='Purchaser_categories')
        col2.write(fig9)

    elif demo == 'Education':
        fig10 = px.histogram(
        data_frame=df,
        x="NumDealsPurchases",
        y="Purchases",
        title="Purchases based on Deals",
        color='Education')
        col2.write(fig10)

    elif demo == 'Marital Status':
        fig8 = px.histogram(
        data_frame=df,
        x="NumDealsPurchases",
        y="Purchases",
        title="Purchases based on Deals",
        color='Marital_Status')
        col2.write(fig8)


with platforms:
    col1a, col2a = st.beta_columns(2)
    col1a.header('V. Customer purchases on Web')
    col1a.subheader('Web Visits vs. Web Purchases')
    col2a.header('VI. Customer purchase based on Deals')
    col2a.subheader('Will low earners accept deal offers more than high earners? ')

    if demo == 'Income':

      figa = px.scatter(
        data_frame=df,
         x="NumWebVisitsMonth",
         y="NumWebPurchases",
         color="Income_categories"
         )
      col1a.write(figa)

    elif demo == 'Education':

       figb = px.scatter(
       data_frame=df,
       x="NumWebVisitsMonth",
       y="NumWebPurchases",
       color="Education"
       )
       col1a.write(figb)


    elif demo == 'Marital Status':

       figc = px.scatter(
       data_frame=df,
       x="NumWebVisitsMonth",
       y="NumWebPurchases",
       color="Marital_Status"
       )
       col1a.write(figc)

    elif demo == 'Purchaser type':

      figd = px.scatter(
      data_frame=df,
      x="NumWebVisitsMonth",
      y="NumWebPurchases",
      color="Purchaser_categories"
      )
      col1a.write(figd)

    elif demo == 'Generation':

       fige = px.scatter(
       data_frame=df,
       x="NumWebVisitsMonth",
       y="NumWebPurchases",
       color="Purchaser_categories"
       )
       col1a.write(fige)

    fig = px.scatter(
    data_frame=df,
    x="Income",
    y="NumDealsPurchases"
    )
    col2a.write(fig)

#################################################
#             MACHINE LEARNING                  #
#################################################

Col_to_keep = ["Year_Birth", "Education", "Income",
               'Recency', 'Response','Purchases','Transactions',
               'AcceptedCmpTotal']
df_imp = df[Col_to_keep]
df_imp.head()


df_imp['Education'] = df_imp['Education'].astype('str')

encode_educate = OneHotEncoder(handle_unknown='ignore')
encode_educate_df = pd.DataFrame(encode_educate.fit_transform(df_imp[['Education']]).toarray())
df_imp = df_imp.join(encode_educate_df)

df_imp.rename(columns = {0: "2n Cycle", 1: "Basic", 2: "Graduation", 3: "Master", 4: "PhD"},inplace = True)

del df_imp['Education']

df_imp.dropna(inplace=True)

X=df_imp.drop(["Response"], axis=1)
y=df_imp.Response

tree = DecisionTreeClassifier(random_state = 1,
                          max_depth=5,
                           min_samples_split = 2,
                             min_samples_leaf =200)
tree.fit(X,y)


with ml:
    st.title('Predict if a customer will respond to a marketing campaign or not')
    st.header('Fill in the below fields with your current customer information')
    st.subheader('Select the customers Year of birth:')
    year = st.slider('Slide to select birth year of customer',min_value=1893, max_value=2005)
    st.subheader('Select the customers approximate Income:')
    income = st.slider('Slide to select Income customer',min_value=3000, max_value=500000)
    st.subheader('Select duration of customers recency:')
    recency = st.slider('Slide to select receny of customer',min_value=0, max_value=100)
    st.subheader('Select customers number of purchases:')
    purchases = st.slider('Slide to select purchases of customer',min_value=0, max_value=3000)
    st.subheader('Select customers customers transactions:')
    trans = st.slider('Slide to select customer transactions',min_value=0, max_value=50)
    st.subheader('How many campaigns did the customer accept?:')
    acc = st.slider('Slide to select number of accepted campaigns',min_value=0, max_value=4)

    ed = st.selectbox('Select Education ', ['Graduation','Master','PhD','2n Cycle','Basic'])

if ed == 'Graduation':
    data1 = {'Year_Birth': year, 'Income': [income],'Recency':[recency],'Purchases':[purchases],
           'Transcations':[trans],'AcceptedCmpTotal':[acc],
           '2n Cycle':[0],'Basic':[0],'Graduation':[1],'Master':[0],'PhD':[0]}
    Xp = pd.DataFrame(data1)
    tree.predict(Xp)

elif ed == 'Master':
    data2 = {'Year_Birth': year, 'Income': [income],'Recency':[recency],'Purchases':[purchases],
           'Transcations':[trans],'AcceptedCmpTotal':[acc],
           '2n Cycle':[0],'Basic':[0],'Graduation':[0],'Master':[1],'PhD':[0]}
    Xp = pd.DataFrame(data2)
    tree.predict(Xp)

elif ed == 'PhD':
    data3 = {'Year_Birth': year, 'Income': [income],'Recency':[recency],'Purchases':[purchases],
           'Transcations':[trans],'AcceptedCmpTotal':[acc],
           '2n Cycle':[0],'Basic':[0],'Graduation':[0],'Master':[1],'PhD':[0]}
    Xp = pd.DataFrame(data3)
    tree.predict(Xp)

elif ed == 'Basic':
    data4 = {'Year_Birth': year, 'Income': [income],'Recency':[recency],'Purchases':[purchases],
           'Transcations':[trans],'AcceptedCmpTotal':[acc],
           '2n Cycle':[0],'Basic':[1],'Graduation':[0],'Master':[0],'PhD':[0]}
    Xp = pd.DataFrame(data4)
    tree.predict(Xp)

elif ed == '2n Cycle':
    data5 = {'Year_Birth': year, 'Income': [income],'Recency':[recency],'Purchases':[purchases],
           'Transcations':[trans],'AcceptedCmpTotal':[acc],
           '2n Cycle':[1],'Basic':[0],'Graduation':[0],'Master':[0],'PhD':[0]}
    Xp = pd.DataFrame(data5)
    tree.predict(Xp)





if st.button('Predict Customer Response'):
    if tree.predict(Xp) == 1:
      st.write('Yes the customer will respond to the marketing campaign')
    else:
      st.write('No the customer will not respond to the marketing campaign')

#tree.predict(Xp)
