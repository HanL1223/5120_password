from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
model = None
vectorizer = None
with open("xgb_model2.pkl", "rb") as f:
    model = pickle.load(f)
with open('vectorizer2.pkl','rb') as f:
    vectorizer = pickle.load(f)
def evalu(text):
    X = vectorizer.transform([text])
    y_pred = model.predict(X)
    return y_pred