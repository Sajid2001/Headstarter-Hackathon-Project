import streamlit as st
import pandas as pd
import pandasql as ps
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


st.title('Headstarter Student Analysis')
df = pd.read_csv('HackathonDataset.csv')

st.header('Full Data')
st.write(df)

st.header('Majors Bar Chart')
major_counts = df['Major'].value_counts()
print(major_counts.head(10))
st.bar_chart(major_counts)

filter_query = "SELECT * FROM df WHERE Complain = 0 AND Headstarter_Rating >= 7"
filtered_data = ps.sqldf(filter_query, locals())
st.header('Filtered Data (Complaints and Low Ratings Removed)')
st.write(filtered_data)

income = st.slider('How much Income?', 10000, 100000)
# dual query of people who are not computer science majors and make less than 100k income
# pyplot - see plots available
# logistic regression
st.header('Incomes Below Six Figures and Other Variables')
major_income_query = f"SELECT * FROM filtered_data WHERE Income < {income}"
income_counts = ps.sqldf(major_income_query, locals())
st.write(income_counts)

figOne = px.scatter(income_counts, x='Minutes_Spent_on_Headstarter', y="Income")
st.plotly_chart(figOne)

figTwo = px.scatter(income_counts, x='Days_Since_Last_Cohort', y="Income")
st.plotly_chart(figTwo)

emails = st.slider('How many emails opened?', 0,20)
st.header("Email Opens and Other Variables")
emails_query = f"SELECT * FROM filtered_data WHERE Email_Opens <= {emails}"
email_opens = ps.sqldf(emails_query, locals())
st.write(email_opens)

figThree = px.scatter(email_opens, x='Email_Opens', y="Minutes_Spent_on_Headstarter")
st.plotly_chart(figThree)


figFive = px.scatter(email_opens, x='Email_Opens', y="Amount_Spent_On_Courses")
st.plotly_chart(figFive)

figSix = px.scatter(email_opens, x='Email_Opens', y="Videos_Watched")
st.plotly_chart(figSix)

figSeven = px.scatter(email_opens, x='Email_Opens', y="Site_Visits_Per_Month")
st.plotly_chart(figSeven)


st.header('Education Filter')
education = st.radio("What's education level do you want to filter through?",("'In-College'", "'High School'", "'Bachelors'", "'Masters'", "'PhD'"))
young_compsci_query = f"SELECT * FROM filtered_data WHERE Education = {education}"
young_compsci = ps.sqldf(young_compsci_query, locals())
st.write(young_compsci)

st.header('MultiQuery')
multi_query = f"SELECT * FROM filtered_data WHERE Education = {education} AND Income < {income}"
multi_query_data = ps.sqldf(multi_query, locals())
st.write(multi_query_data)


#Linear Regression
y = filtered_data['Probability_Of_Getting_Offer']
offer_cols = ['Highest_Leaderboard_Rank', 'Minutes_Spent_Coding', 'Questions_Completed','Minutes_Spent_on_Headstarter']
X = filtered_data[offer_cols]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0) 
linreg =  linear_model.LinearRegression() 
linreg.fit(X_train,y_train) 
y_pred=linreg.predict(X_test)
r_squared = linreg.score(X, y)

print(r_squared)
error = mean_squared_error(y_test, y_pred)
print(error)

st.header('Machine Learning Model Results (Linear Regression)')
st.subheader(f'R Squared: {r_squared}')
st.caption('How accurate each data point is. The closer to 1, the more accurate.')
st.subheader(f'Mean Error Squared: {error}')
st.caption('Error between each data point and the expected.')

#Google Neural Networks and Decision Trees