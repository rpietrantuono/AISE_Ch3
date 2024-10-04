import sklearn.metrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline

from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
import pandas as pd

df = pd.read_csv('./bank.csv', delimiter=';', decimal=',')

# Assume there was some EDA and feature analysis to select below
#feature_cols = ['job', 'marital', 'education', 'contact', 'housing', 'loan', 'default', 'day']

# Features and target
X = df[:].copy()
y = df['y'].apply(lambda x: 1 if x == 'yes' else 0).copy()
print(X, y)
# Train/test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

numeric_features = ['age', 'balance']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['job', 'marital', 'education', 'contact', 'housing', 'loan', 'default','day']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


# Add classifier to the preprocessing pipeline
clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression())])


pipe = clf_pipeline.fit(X_train, y_train)
print(pipe)

predictions = clf_pipeline.predict(X_test)
print("Accuracy", sklearn.metrics.accuracy_score(y_test,predictions))
print("Classification report", sklearn.metrics.classification_report(y_test,predictions))