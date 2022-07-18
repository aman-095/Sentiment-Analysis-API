from fastapi import FastAPI

import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
def preprocess(token):
    
    token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
    token = re.sub("(@[A-Za-z0-9_]+)", "", token)
    token = re.sub("(#[A-Za-z0-9_]+)", "", token)
    return token

app = FastAPI()
@app.get("/{text}")
async def home(text: str):
    sentiment = preprocess(text)
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=joblib.load(open('feature.pkl', "rb")))
    
    sentiment_test = transformer.fit_transform(loaded_vec.fit_transform(np.array([sentiment])))
    clf = SVC()
    
    
    
    clf = joblib.load('classifier.joblib.pkl')
    y_pred = clf.predict(sentiment_test.reshape(1, -1))
    
    return {"sentiment":y_pred[0]}

    
    
