# About
Topic Modelling in TF-IDF and LDA

by August Erséus and Ted Strömfelt


# How to run

## Notebooks
Change this cell to your folder location
```python
import sys
sys.path.insert(0,'/Users/august/Coding/topic-modelling')
```

## Data
### Million Headlines dataset
Load data like so:
```python
from topic_modelling.data_loader import DataLoader
dl = DataLoader()
corpus = dl.load(DATA_PATH, n_rows=N_ROWS, lemmatize=True, stemming=True, data_column=DATA_COLUMN)
```

where ``DATA_PATH`` is either a .csv or a presaved ``pd.DataFrame`` in .pickle, and ``DATA_COLUMN`` is the column name for your text data.

### 20 Newsgroups dataset

```python
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

# chose categories, or set categories = "all"
categories = ['rec.autos', 'talk.religion.misc',
              'comp.graphics', 'sci.space', 'talk.politics.guns']

newsgroups_train = fetch_20newsgroups(subset='train',
                                      categories=categories, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', 
                                      categories=categories,  remove=('headers', 'footers', 'quotes'))
                                      
data_df = pd.DataFrame(newsgroups_train, columns=["data", "target"])
test_data_df = pd.DataFrame(newsgroups_test, columns=["data", "target"])

dl = DataLoader()

corpus = dl.load_df(data_df)
test_corpus = dl.load_df(test_data_df)
```

## TF-IDF
```python
from topic_modelling.TFIDF import TFIDFVectorizer

TFIDF = TFIDFVectorizer()
TFIDF_matrix = TFIDF.fit(corpus)
```
where ``corpus`` is a ``pd.DataFrame`` with just one column, consisting of the tokenized words for each document.


## k-means
```python

from topic_modelling.kmeans import KMeansCluster

n_clusters = 5

cluster = KMeansCluster(corpus, TFIDF_method="native", num_clusters=n_clusters)
doc_clusters, cluster_assignments = cluster.cluster_documents(verbosity=False)
```
where ``corpus`` is a ``pd.DataFrame`` with just one column, consisting of the tokenized words for each document.

## LDA
Fit model:
```python
lda.fit(verbosity=True, num_topics=9, iterations=50)
```
Plot wordclouds:
```python
lda.plot_word_clouds_all()
```
Get words with highest probability from each topic:
```python
topicswords, weights = lda.get_topics()
```
Predict an unseen sentence:
```python
sentence = "farmer sentenced to prison"
data = dl.process_data(sentence)
lda.predict_topic(data)
```

