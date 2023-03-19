from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import nltk.data
import numpy as np
tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')


def sentence_splitter(context,length=384):
    sentence_list=[]
    for x in context:   
        temp=''
        count=0
        for k in tokenizer.tokenize(x):
            count+=len(k)
            if count>length:
               sentence_list.append(temp)
               count=len(k)
               temp=k
               continue
            temp+=k
    return sentence_list        
            
def tf_idf_similarity_check(query,sentence_list):
    combined=[query]+sentence_list
    tfidfvectorizer = TfidfVectorizer(stop_words='english',sublinear_tf=True, strip_accents='unicode',analyzer='word',token_pattern=r'\w{2,}',ngram_range=(1, 4),max_features=50000)
    tfidfvectorizer_fit=tfidfvectorizer.fit(combined)
    combined_tf_idf=tfidfvectorizer_fit.transform(combined)
    
    cosine_sim=linear_kernel(combined_tf_idf[0:1],combined_tf_idf).flatten()
    index=cosine_sim.argsort()[-2]
    return cosine_sim[index],combined[index]

if __name__=='__main__':
    df=pd.read_excel("/home/htd/Downloads/qa_test.xlsx",engine='openpyxl',index_col=0)
    query='What is the cause of the tides'
    context=list(set(df["context"].values.tolist()))    
    sentence_list=sentence_splitter(context)
    sim_score,sim_para=tf_idf_similarity_check(query,sentence_list)            
    #print(sim_score,sim_para)
