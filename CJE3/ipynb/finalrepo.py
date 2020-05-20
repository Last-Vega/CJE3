#---------------------------------------------
# 1. 初期設定
#---------------------------------------------

### A. ライブラリの読み込み
import os
from janome.tokenizer import Tokenizer
import re
import math
import pandas as pd

# 分かち書きクラスの初期化
t = Tokenizer()

### B. フォルダやファイル名の設定

# 索引ファイルの指定
INDEX = "../index"
index_file = INDEX + "/index3.txt"

### C. 各種辞書オブジェクトの初期化

# 索引語idf値の辞書オブジェクト
idf_scores = {}
# 索引語tfidf値の辞書オブジェクト
tfidf_scores = {}
# 検索語tf値の辞書オブジェクト
query_tf = {}
# 検索語tfidf値の辞書オブジェクト
query_tfidf = {}
# 不要語辞書オブジェクト
stopwords = {}
# 検索語の辞書オブジェクト
query_words = {}
# 順位付け対象文書用辞書オブジェクト
ranking_docs = {}

#---------------------------------------------
# 2. 索引ファイルの読み込み
#---------------------------------------------

# 索引ファイルを読み込みモードで開く
f = open(index_file, 'r')

# 索引ファイルを1行づつ処理
for line in f:
    ### A. 行末処理と要素の分割

    # 行末の改行文字除去
    line = line.rstrip()
    # 行の分割
    split_line = line.split("\t")

    # 配列要素の取得
    word = split_line[0] # 索引語
    doc = split_line[1] # ファイル名
    idf = float(split_line[3])
    tfidf = float(split_line[4])


    ### B. 重み値の取得
    idf_scores[word] = idf
    # 辞書オブジェクトへの代入（改良版）
    if word in tfidf_scores:
        tfidf_scores[word][doc] = tfidf

    else:
        tfidf_scores[word] = {}
        tfidf_scores[word][doc] = tfidf

### C. tfidf_scoresからデータフレームを作成
tfidf_table = pd.DataFrame(tfidf_scores)

# NaNを0に置き換える
tfidf_table = tfidf_table.fillna(0)

#---------------------------------------------
# 3. 検索質問の処理
#---------------------------------------------

# 検索質問
query = '吾輩は猫である'
# 検索質問データフレーム用ファイル名
query_file = 'query'

### A. 不用語削除ルールの定義

# 不要語としてマッチしたいパターンの定義
pattern = re.compile(r"^[　-ー]$")
# 不要語の追加
stopwords['という'] = 1
stopwords['にて'] = 1

### B. 検索質問の分かち書き
tokens = t.tokenize(query)

# 分かち書きされた語オブジェクトの処理
for token in tokens:
    ### C. 不用語処理
    # 正規表現
    if pattern.match(token.surface):
        continue

    # 不用語リスト
    if token.surface in stopwords:
        continue

    #  検索語の追加とtf値の加算
    query_words[token.surface] = 1
    """
    for query_word in sorted(query_words):
        for doc in tfidf_scores[query_word]:
            ranking_docs[doc] = 1
    """

### D. 検索語の重み付け

# 索引語を一つずつ処理
for word in idf_scores:
    # 索引語をキーとした検索質問tf値用とtfidf値用オブジェクトを初期化
    query_tf[word] = {}

    # キー：索引語、擬似文書名「query」、値：0
    query_tf[word][query_file] = 0

# 検索語を一つずつ処理
for query_word in query_words:
    # 検索語==索引語なレコードのtf値をquery_wordsから代入
    for index_word in idf_scores:
        if query_word == index_word:
            query_tf[index_word][query_file] += 1

# 検索語の出現頻度（`tf`値）と`idf`値を使って、`tfidf`値を算出する
for word in query_tf:
    for q in query_tf[word]:
        tf = query_tf[word][q]
        idf = idf_scores[word]
        tfidf = tf * idf
        query_tfidf[word] = tfidf

### D. query_tfidfからデータフレームを作成
query_table = pd.DataFrame(query_tfidf, index=['query',])

#---------------------------------------------
# 4. 類似度の計算と順位付け
#---------------------------------------------

### A. 順位付け対象文書の同定
for query_word in query_words:
    for doc in tfidf_scores[query_word]:
        # 値`1`で初期化
        ranking_docs[doc] = 1

# 対象文書を一つずつ処理
for doc in ranking_docs:
    ### B. 余弦関数の分子の算出

    # 分子変数の初期化
    numerator = 0
    # 文書データフレームからdocにマッチする行のデータを取得
    doc_vec = tfidf_table.loc[doc]
    # 検索質問データフレームからquery_fileにマッチする行のデータを取得
    query_vec = query_table.loc['query']
    # 検索質問ベクトルの要素を添字を使って巡回
    for i in range(len(query_vec.values)):
        # query_vecとdoc_vecのi番目の要素を掛け合わせて、分子変数に足していく
        i_value = query_vec.values[i] * doc_vec.values[i]
        numerator += i_value

    ### C. 余弦関数の分母の算出

    # 分母変数の初期化
    denominator = 0
    # 検索語ベクトル積の変数の初期化
    query_value = 0
    # 索引語ベクトル積の変数の初期化
    doc_value = 0
    # 検索質問ベクトルの要素を添字を使って巡回
    for i in range(len(query_vec.values)):
        # 検索質問ベクトルのi番目の値の二乗をquery_valueに加算していく
        query_value += query_vec.values[i] ** 2
        # 文書ベクトルのi番目の値の二乗をdoc_valueに加算していく
        doc_value += doc_vec.values[i] ** 2

    # query_valueとdoc_valueの平方根を掛け合わせて分母とする
    denominator = math.sqrt(query_value) * math.sqrt(doc_value)

    ### D. 余弦関数の算出
    cosine = numerator / denominator

    # ranking_docsへ値を代入
    ranking_docs[doc] = cosine

#---------------------------------------------
# 5. 順位付け結果の出力
#---------------------------------------------

# 類似度の高い順に文書を表示
print(sorted(ranking_docs.items(), key=lambda x:x[1], reverse=True))
