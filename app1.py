import pandas
import pickle
import numpy as np
import streamlit as st
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def load_data(x):
    raw = pandas.read_csv(x, encoding='utf-8', dtype='str', keep_default_na=False, na_values=[''])
    data = raw.copy()
    data = data.dropna()
    data = data.reset_index()
    data = data.drop(['index', 'education'], axis=1)

    X = data.drop(['TenYearCHD'], axis=1)
    X = X.apply(LabelEncoder().fit_transform)
    y = data['TenYearCHD']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)

    training_accuracy = logreg.score(X_train, y_train)
    testing_accuracy = logreg.score(X_test, y_test)
    return (logreg)

data_train_state = st.text('Loading and training data.....')
logreg = load_data("https://raw.githubusercontent.com/krao14/krao14repo/main/framingham_heart_disease.csv")
data_train_state.text('Done: Loading and training data')

data = {"model": logreg}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)

with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]

st.title("Heart Disease Prediction")

st.write("""### We need some information to predict heart disease""")

male = (
    0,
    1
)

age = st.slider("Choose the age", min_value=0, max_value=80)

currentSmoker = (
    0,
    1
)

cigsPerDay = st.slider("Cigarettes per day", min_value=0, max_value=30)

BPMeds = (
    0,
    1
)

prevalentStroke = (
    0,
    1
)

diabetes=(
    0,
    1
)

totChol = st.slider("Total Cholestrol", min_value=100, max_value=300)

male = st.selectbox("Male", male)
currentSmoker = st.selectbox("Current Smoker", currentSmoker)
BPMeds=st.selectbox("BP Medication", BPMeds)
prevalentStroke=st.selectbox("Prevalent Stroke", prevalentStroke)
diabetes=st.selectbox("Diabetes", diabetes)

ok = st.button("Predict heart disease")
if ok:
    x = np.array([[male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, 0,diabetes, totChol,1,1,0,0,0]])
    y_pred = regressor.predict(x)
    if str(y_pred[0]) == '0':
        st.subheader("Safe: No heart disease predicted")
    else:
        st.subheader("Alert: Predicted heart disease")
