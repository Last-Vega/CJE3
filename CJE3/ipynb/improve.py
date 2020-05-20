def read_file(index_file):
    f = open(index_file, 'r')
    idf_scores = {}
    tfidf_scores = {}
    for line in f:
        line = line.rstrip()
        split_line = line.split("\t")
        word = split_line[0]
        doc = split_line[1]
        idf = float(split_line[3])
        tfidf = float(split_line[4])

        idf_scores[word] = idf
        if word in tfidf_scores:
            tfidf_scores[word][doc] = tfidf
        else:
            tfidf_scores[word] = {}
            tfidf_scores[word][doc] = tfidf
    return idf_scores, tfidf_scores

def deal_with_query(query, tfidf_scores):
    stopwords = {}
    query_words = {}
    ranking_docs = {}
    pattern = re.compile(r"^[　-ー]$")
    stopwords['という'] = 1
    stopwords['にて'] = 1
    tokens = t.tokenize(query)
    for token in tokens:
        if pattern.match(token.surface):
            continue
        if token.surface in stopwords:
            continue
        if token.surface in query_words:
            query_words[token.surface] += 1
        else:
            query_words[token.surface] = 1
    return query_words, ranking_docs

def query_weighting(idf_scores, query_file, query_words):
    query_tf = {}
    query_tfidf = {}
    for index_word in idf_scores:
        query_tf[index_word] = {}
        query_tf[index_word][query_file] = 0
    for query_word in query_words:
        for index_word in idf_scores:
            if query_word == index_word:
                query_tf[query_word][query_file] = query_words[query_word]
    for query_word in query_tf:
        for doc in query_tf[query_word]:
            tf = query_tf[query_word][doc]
            idf = idf_scores[query_word]
            tfidf = tf * idf
            query_tfidf[query_word] = tfidf
    return query_tfidf

def create_dataframe(tfidf_scores, query_tfidf):
    tfidf_table = pd.DataFrame(tfidf_scores)
    tfidf_table = tfidf_table.fillna(0)
    query_table = pd.DataFrame(query_tfidf, index=['query',])
    return tfidf_table, query_table

def identify_docs(query_words, tfidf_scores, ranking_docs):
    try:
        for query_word in query_words:
            for doc in tfidf_scores[query_word]:
                ranking_docs[doc] = 1
    except KeyError:
        print("Please input Japanese sentences.")
        exit()
    return ranking_docs

def cosine_similality(ranking_docs, tfidf_table, query_table):
    for doc in ranking_docs:
        doc_vec = tfidf_table.loc[doc]
        query_vec = query_table.loc['query']
        numerator = np.matmul(doc_vec.values, query_vec.values)
        denominator = np.linalg.norm(query_vec.values) * np.linalg.norm(doc_vec.values)
        cosine = numerator / denominator
        ranking_docs[doc] = cosine
    return ranking_docs

def output(ranking_docs):
    print(sorted(ranking_docs.items(), key=lambda x:x[1], reverse=True))

def do_func():
    idf_scores = read_file(index_file)[0]
    tfidf_scores = read_file(index_file)[1]
    query_words = deal_with_query(query, tfidf_scores)[0]
    if not query_words:
        print("Please input not a letter but words or sentences")
        exit()
    ranking_docs = deal_with_query(query, tfidf_scores)[1]
    query_tfidf = query_weighting(idf_scores, query_file, query_words)
    tfidf_table = create_dataframe(tfidf_scores, query_tfidf)[0]
    query_table = create_dataframe(tfidf_scores, query_tfidf)[1]
    ranking_docs = identify_docs(query_words, tfidf_scores, ranking_docs)
    ranking_docs = cosine_similality(ranking_docs, tfidf_table, query_table)
    output(ranking_docs)

#---------------------------------------------
# プログラム本体
#---------------------------------------------
import os
from janome.tokenizer import Tokenizer
import re
import math
import numpy as np
import pandas as pd
t = Tokenizer()

INDEX = "../index"
index_file = INDEX + "/index3.txt"
query = ""
while True:
    input_words = str(input("Query ->"))
    if not input_words:
        break
    else:
        query += "".join(input_words)
print(query)
query_file = 'query'
do_func()
