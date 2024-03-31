#!/usr/bin/env python3
# coding=utf-8

import sys
ID = int(sys.argv[1])
V = int(sys.argv[2])

# %%
import PyPDF2
import os.path
import gzip
import pickle
import sys
import re
import numpy as np
from gensim.utils import simple_preprocess, RULE_DISCARD, RULE_DEFAULT

def count_occurence_in_corpus(corpus, pattern, flags=re.IGNORECASE):
    p = re.compile(pattern, flags=flags)
    return sum((sum((1 if p.match(w) else 0 for w in s)) for s in corpus))

substitute_femme = ['Eve', 'Sara', 'Agar', 'Rebecca', 'Lea', 'Rachel',
    'Tamar', 'Jezabel', 'Elizabeth', 'Rahab', 'Sheera', 'Yehosheeba',
    'Esther', 'Jeanne', 'Salome', 'Debora', 'Yael', 'Dalila', 'Naomi',
    'Ruth', 'Abigail', 'Bath-cheba', 'Myriam', 'Dina', 'Anne', 'Marie',
    'Marthe', 'elisabeth', 'semme', 'femmes', 'femme']


substitute_femme = set(sum((simple_preprocess(w, deacc=True) for w in substitute_femme), start=[]))

substitute_homme = [
    "David", "Moïse", "Abraham", "Isaac", "Jacob", "Joseph", "Samuel", "Paul", "Pierre", "Jean",
    "Élie", "Ésaü", "Noé", "Aaron", "Salomon", "Daniel", "Matthieu", "Josué", "Éliezer", "Benjamin",
    "Ézéchiel", "Job", "Ézra", "Zacharie", "Jacob", "Gédéon", "Caleb", "Adam", "Nathan", "Jérémie",
    "Ézéchias", "Abel", "Éphraïm", "Ruben", "Siméon", "Caïn", "Énoch", "Eliezer", "Ézéchiel", "Boaz",
    "Élie", "Amos", "Osée", "Nahum", "Obed", "Amos", "Éliab", "Josué", "Éphraïm", "Saül", "Seth",
    "Tobie", "Élie", "Manassé", "Siméon", "Esdras", "Jonathan", "Asa", "Joël", "Jérémie", "Joachim",
    "Habacuc", "Siméon", "Uzziel", "Jotham", "Malachie", "Michée", "Moïse", "Joachim", "Joas",
    "Obadiah", "Baruch", "Ézéchiel", "Zacharie", "Élisée", "Esaïe", "Jésus", "Seth", "Zacharie",
    "Tobie", "Manassé", "Azaria", "Jotham", "Nehemiah", "Manassé", "Azariah", "Ézéchias", "Éliezer",
    "Abijah", "Ézéchias", "Salomon", "Zacharie", "Siméon", "Amos", "Ézéchiel", "Ruben", "Asa",
    "Azariah", "Tobie", "Josias", "hommes", "homme"
]

substitute_homme = set(sum((simple_preprocess(w, deacc=True) for w in substitute_homme), start=[]))

def extract_text_from_pdf(pdf_path):
    if os.path.exists(f"{pdf_path}.text.pickle.gz"):
        with gzip.open(f"{pdf_path}.text.pickle.gz", "rb") as f:
            return pickle.load(f)

    with open(pdf_path, 'rb') as file:
        # Utilisez PdfReader à la place de PdfFileReader
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        # Utilisez len(reader.pages) au lieu de reader.numPages
        for page_num in range(len(pdf_reader.pages)):
            if page_num % 100 == 0:
                print(f'Page {page_num}')
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    with gzip.open(f"{pdf_path}.text.pickle.gz", "wb") as f:
        return pickle.dump(text, f)
    return text

files = {
    "code_civil1": 'corpus_neutre_et_bible/code-civil.pdf',
    "code_travail": 'corpus_neutre_et_bible/code-du-travail.pdf',
    "code_famille": 'corpus_neutre_et_bible/code-de-laction-sociale-et-des-familles.pdf',
    "code_educ": 'corpus_neutre_et_bible/Code-de-léducation.pdf',
    "dico": 'corpus_neutre_et_bible/dico_academie_francaise.pdf',
    "bible": 'corpus_neutre_et_bible/Bible_de_Jerusalem_le_bon.pdf'
}

text = dict()
for k, v in files.items():
    text[k] = extract_text_from_pdf(v)

from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess

def text_to_corpus(text):
    text = text.replace('œ', 'oe')
    text = text.replace('Œ', 'oe')
    sentences = sent_tokenize(text)
    corpus_token = []
    for s in sentences:
        sx = []
        for w in simple_preprocess(s, deacc = True):
            if 'µ' in w or '_' in w or 'º' in w:
                continue
            if w in substitute_homme:
                w = "HOMME" # upper case to track subtitution
            if w in substitute_femme:
                w = "FEMME" # upper case to track subtitution
            sx.append(w)
        corpus_token.append(sx)
    return corpus_token

corpus = dict()
for k, v in text.items():
    corpus[k] = text_to_corpus(v)

corpus_neutre = sum((corpus[k] for k in {"code_civil1", "code_famille", "code_educ", "code_travail", "dico"}), start=[])

# %%
from gensim.models import Word2Vec

# %% [markdown]
# **Question** : Est-ce qu'on évalue séparément le stabilité du modèle entraîné uniquement sur le corpus neutre et la stabilité du modèle entraîné sur le corpus neutre et la bible ? Ou on évalue direct la stabilité du modèle entraîné sur le corpus neutre et la bible ? 

# %%
import random
import os
import struct

def seed64():
    return struct.unpack("Q", os.urandom(8))[0]
def seed31():
    return abs(struct.unpack("i", os.urandom(4))[0])

random.seed(seed64())

#corpus = corpus_neutre+corpus["bible"]
corpus = corpus_neutre

#words, wcounts = np.unique(np.concatenate(corpus), return_counts=True)
#x = list(zip(words, wcounts))
#x.sort(key=lambda x: x[1])
#with open("words.txt", "w") as fout:
#    for w,c in x:
#        fout.write(f"{w} {c}\n")

window_size = 5
vector_size = V

random.shuffle(corpus)
model = Word2Vec(window = window_size, vector_size = vector_size, seed=seed31(), compute_loss=True, alpha=0.025)

# Keep word only if they apear at less twise
model.build_vocab(corpus, trim_rule=lambda w, c, x: RULE_DEFAULT if c>1 else RULE_DISCARD)
fout = open(f"xrun4/run-G{ID:03d}W{window_size:03d}V{vector_size:03d}.loss.txt", "w")
for k in range(200):
    model.train(corpus, total_examples=model.corpus_count, epochs=10, compute_loss=True, start_alpha=0.025*(1000-k)/1000, end_alpha=0.025*(1000-k)/1000)
    print(f"loss E{k:03d} =", model.get_latest_training_loss())
    fout.write(str(model.get_latest_training_loss())+"\n")
    fout.flush()
    print("femme=",{x[0] for x in model.wv.most_similar("FEMME", topn=10)})
    print("homme=",{x[0] for x in model.wv.most_similar("HOMME", topn=10)})
    if (k+1)%20:
        model.wv.save(f"xrun4/run-G{ID:03d}W{window_size:03d}V{vector_size:03d}E{(k+1)*10:06d}.word2vec")
model.wv.save(f"xrun4/run-G{ID:03d}W{window_size:03d}V{vector_size:03d}E{(k+1)*10:06d}.word2vec")

