from __future__ import annotations
import sqlite3
import json
from pathlib import Path
import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tripmind.text import tokens
from .common import write_json


def build_index(folder: Path, max_features: int = 120000) -> dict:
    connection = sqlite3.connect(folder/"documents.sqlite")
    def texts():
        for (value,) in connection.execute("SELECT metadata FROM documents ORDER BY row_idx"):
            doc=json.loads(value)
            yield " ".join([doc["title"],doc["city"],doc["section"],doc["text"]])
    vectorizer = TfidfVectorizer(tokenizer=tokens,token_pattern=None,lowercase=False,
        max_features=max_features,dtype=np.float32,sublinear_tf=True,norm="l2")
    try:
        matrix = vectorizer.fit_transform(texts())
    finally:
        connection.close()
    sparse.save_npz(folder/"tfidf.npz",matrix,compressed=True)
    joblib.dump(vectorizer,folder/"vectorizer.joblib",compress=3)
    info = {"kind":"local_sparse_tfidf","rows":matrix.shape[0],"features":matrix.shape[1],
        "nonzero_values":matrix.nnz,"semantic_embeddings":False,"tokenization":"Latin words + simplified-Chinese bigrams",
        "source_filter":"city/country filter before cosine ranking"}
    write_json(folder/"index.json",info)
    return info
