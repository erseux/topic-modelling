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
Load data like so:
```python
from topic_modelling.data_loader import DataLoader
dl = DataLoader()
corpus = dl.load(DATA_PATH, n_rows=N_ROWS, lemmatize=True, stemming=True, data_column=DATA_COLUMN)
```

where ``DATA_PATH`` is either a .csv or a presaved ``pd.DataFrame`` in .pickle, and ``DATA_COLUMN`` is the column name for your text data.
