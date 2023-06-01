import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.inspection import PartialDependenceDisplay

def pdp_visualize(model, title, X_train_scaled, f_names, y_train):
    X_scaled_df= pd.DataFrame(X_train_scaled, columns = f_names)
    model.fit(X_scaled_df, y_train)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_title(title)
    PartialDependenceDisplay.from_estimator(model, X_scaled_df, X_scaled_df.columns, ax=ax)
    fig.savefig(title)

df_train = pd.read_csv('archive/Paitients_Files_Train.csv')
df_test = pd.read_csv('archive/Paitients_Files_Test.csv')

y = df_train['Sepssis'].map({'Positive':1, 'Negative': 0})
X = df_train.drop(['ID','Sepssis'], axis='columns')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2023)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=10000000000),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Random Forest': RandomForestClassifier(),
    'GaussianNB': GaussianNB(),
    'KNeighbors': KNeighborsClassifier(),
    'Neural Network': MLPClassifier(max_iter=10000000)
}

model_names = []
model_average_scores = []
for model_name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train)
    model_names.append(model_name)
    model_average_scores.append(scores.mean())
    
df_model = pd.DataFrame()
df_model['model'] = model_names
df_model['average score'] = model_average_scores

clf = GradientBoostingClassifier()
clf.fit(X_train_scaled, y_train)
y_predicted = clf.predict(X_test_scaled)
print(classification_report(y_test, y_predicted))
model = MLPClassifier(max_iter=10000000)
f_names = ["Plasma glucose (PRG)", "Blood Work Result-1 (PL)", "Blood Pressure (PR)", "Blood Work Result-2 (SK)", "Blood Work Result-2 (TS)", "Body mass index (M11)", "Blood Work Result-4 (BD2)","Age", "Has Insurance"]
pdp_visualize(model, "Neural Network", X_train_scaled, f_names, y_train)