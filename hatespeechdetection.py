import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk 
import tkinter as tk
nltk.download('punkt')
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

nltk.download('stopwords')
nltk.download('plunkt')
nltk.download('wordnet')
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
train_data.head()
def delete_similar(inp, pat):
    r = re.findall(pat, inp)
    for i in r:
        inp = re.sub(i, '', inp)
    return inp
root = tk.Tk()
canvas1=tk.Canvas(root,width=400,height=550, bg = "Yellow")
canvas1.pack()
labelT = tk.Label(root, text='WELCOME TO THE HATE SPEECH DETECTOR ')
canvas1.create_window(200, 50, window=labelT)


# Gender
label2 = tk.Label(root, text=' TWEET: ')
canvas1.create_window(100, 130, window=label2)

entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 130, window=entry2)
def analysis():
    global ID #our 1st input variable
    ID = "1234567"
    
    global TWEET	 #our 2nd input variable
    TWEET	 = str(entry2.get()) 
    train_data = pd.read_csv("train.csv")

    global STATUS
    STATUS = "0"
    a=[ID,TWEET]
    test_data.loc[len(train_data.index)] = a
    print(test_data.tail(1))
    train_data['Data_without_user']=np.vectorize(delete_similar)(train_data['tweet'],"@[\w]*")
    train_data['Data_without_user'] = train_data['Data_without_user'].apply(lambda x: x.lower())
    train_data['Data_without_user'] = train_data['Data_without_user'].apply(lambda x: re.sub(r'[^a-zA-Z]',' ',x))
    train_data['tokenized_data'] = train_data['Data_without_user'].apply(lambda x: word_tokenize(x))
    stop_words = set(stopwords.words('english'))
    train_data['Data_without_stopwords'] = train_data['tokenized_data'].apply(lambda x: [i for i in x if not i in stop_words])
    stemming = PorterStemmer()
    train_data['stemmed_data'] = train_data['Data_without_stopwords'].apply(lambda x:' '.join([stemming.stem(i) for i in x]))
    lemmatizing = WordNetLemmatizer()
    train_data['lemmatized_data'] = train_data['Data_without_stopwords'].apply(lambda x:' '.join([lemmatizing.lemmatize(i) for i in x]))
    vectoriser = CountVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')
    trainstem = vectoriser.fit_transform(train_data['stemmed_data'])
    trainlem = vectoriser.fit_transform(train_data['lemmatized_data'])
    trainstem.toarray()
    trainlem.toarray()
    tfidf_vectoriser = TfidfVectorizer(max_df=0.90,min_df=2,max_features=1000,stop_words='english')
    traintfidfstem = tfidf_vectoriser.fit_transform(train_data['stemmed_data'])
    traintfidflem = tfidf_vectoriser.fit_transform(train_data['lemmatized_data'])
    x = traintfidfstem
    y = train_data['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    svc1=SVC()
    svc1.fit(x_train,y_train)
    predict_svc1=svc1.predict(x_test)
    test_data['Data_without_user'] = np.vectorize(delete_similar)(test_data['tweet'], "@[\w]*")
    test_data['Data_without_user'] = test_data['Data_without_user'].apply(lambda x: x.lower())
    test_data['Data_without_user'] = test_data['Data_without_user'].apply(lambda x: re.sub(r'[^\w\s]',' ', x))
    test_data['Data_without_user'] = test_data['Data_without_user'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ', x))
    test_data['Data_without_user'] = test_data['Data_without_user'].apply(lambda x: re.sub(r'[^a-zA-Z]',' ',x))
    test_data['tweet_token'] = test_data['Data_without_user'].apply(lambda x: word_tokenize(x))
    test_data['Data_without_stopwords'] = test_data['tweet_token'].apply(lambda x: [i for i in x if not i in stop_words])
    test_data['tweet_stemmed'] = test_data['Data_without_stopwords'].apply(lambda x: ' '.join([stemming.stem(i) for i in x]))
    test_data['lemmatized_data'] = test_data['Data_without_stopwords'].apply(lambda x: ' '.join([lemmatizing.lemmatize(i) for i in x]))
    vec_bow = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    test_bs = vec_bow.fit_transform(test_data['tweet_stemmed'])
    test_bl = vec_bow.fit_transform(test_data['lemmatized_data'])
    tfidfvec = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    testtfidf_stem = tfidfvec.fit_transform(test_data['tweet_stemmed'])
    testtfidf_lemm = tfidfvec.fit_transform(test_data['lemmatized_data'])
    test_svc=svc1.predict(testtfidf_lemm)
    print(test_svc[-1])
    if test_svc[-1]==1:
        Prediction_result = 'Hate speech detected'
    else:
        Prediction_result = 'data is free of hate speech'
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(260, 500, window=label_Prediction)
button1 = tk.Button (root, text='CHECK FOR HATE SPEECH',command=analysis, bg='orange') # button to call the 'values' command above 
canvas1.create_window(270, 460, window=button1)        
root.mainloop()