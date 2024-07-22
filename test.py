import pandas as pd
import numpy as np
from ast import literal_eval
import os
from openai import OpenAI
from dotenv import load_dotenv
from os import getenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

OPENAI_MODEL="gpt-3.5-turbo"

load_dotenv()
# api key and org id are used by default if named correctly
client = OpenAI(project=os.getenv('OPENAI_PROJECT_ID'))
df = pd.read_csv('labeled_embedded_sentences.csv')
df['embedding'] = df.embedding.apply(literal_eval).apply(np.array)  # convert string to array
#print(df.head())
# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    list(df.embedding.values), list(df.topic.values), test_size=0.2, random_state=42
)

# train random forest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
probas = clf.predict_proba(X_test)

report = classification_report(y_test, preds)
print(report)
print("done")