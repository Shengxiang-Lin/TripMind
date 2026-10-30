"""City-filtered cosine search over local TF-IDF vectors, with source metadata."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from functools import lru_cache
import joblib
from scipy import sparse
from .text import profile_query


class VectorKnowledgeRetriever:
    def __init__(self, folder: str | Path):
        self.folder=Path(folder)
        # Only artifacts generated and checksummed by our local pipeline are loaded.
        self.vectorizer=joblib.load(self.folder/"vectorizer.joblib")
        self.matrix=sparse.load_npz(self.folder/"tfidf.npz")
        with sqlite3.connect(self.folder/"documents.sqlite") as db:
            self.metadata=db.execute("SELECT row_idx,city,country,language,doc_id FROM documents ORDER BY row_idx").fetchall()
        if len(self.metadata)!=self.matrix.shape[0]:
            raise ValueError("Knowledge metadata/vector row mismatch")

    def retrieve_documents(self, profile, top_k: int = 5) -> list[dict]:
        if top_k<=0: return []
        eligible=[r for r in self.metadata if r[1] in {profile.city,"*"} and r[2] in {profile.country,"*",""}]
        if not eligible: return []
        indices=[r[0] for r in eligible]
        query=self.vectorizer.transform([profile_query(profile)])
        scores=(self.matrix[indices]@query.T).toarray().ravel()
        ranked=sorted(zip(eligible,scores),key=lambda x:(x[0][1]==profile.city,float(x[1]),x[0][3]=="zh",x[0][4]),reverse=True)
        docs=[]
        with sqlite3.connect(self.folder/"documents.sqlite") as db:
            for meta,score in ranked:
                if score<=0: continue
                doc=json.loads(db.execute("SELECT metadata FROM documents WHERE row_idx=?",(meta[0],)).fetchone()[0])
                doc["score"]=float(score); docs.append(doc)
                if len(docs)==top_k: break
        return docs

    def retrieve(self, profile, top_k: int = 5) -> list[str]:
        return [d["text"] for d in self.retrieve_documents(profile,top_k)]


@lru_cache(maxsize=2)
def load_vector_retriever(folder: str) -> VectorKnowledgeRetriever:
    return VectorKnowledgeRetriever(folder)
