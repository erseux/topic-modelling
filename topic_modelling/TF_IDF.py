from topic_modelling.data_loader import DataLoader
import pandas as pd
import numpy as np
import math
import collections
import tqdm

class TFIDFVectorizer:
    def fit(self, corpus: pd.DataFrame, max_df=1.0):
        """
        Generates a TFIDF matrix from a corpus with processed text.
        input:
            Corpus: pd.DataFrame or pd.Series with a single column containing
                documents in the shape of lists of words (List[str])
            max_df: float - the Document Frequency threshold for sorting out
                common words
        return:
            a numpy matrix with TFIDF vectors for all documents
        """
        self.corpus = corpus

        self.N_D = len(self.corpus)
        
        self.DF_THRESHOLD = max_df

        self.doc_freq = {}

        self.TF_list = self.frequencies()

        self.filter_tokens()

        self.unique_words = set(self.doc_freq.keys())   # Sätts efter att vi filtrerat ut tokens

        self.N_W = len(self.unique_words)   # Storleken efter filtrering 

        self.vocab_w_i = {j:i for i,j in enumerate(sorted(self.doc_freq.keys()))}
        self.vocab_i_w = list(self.vocab_w_i.keys())

        self.TF_IDF_matrix = self.TF_IFD()
        print("fitted TF-IDF matrix")
        return self.TF_IDF_matrix

    def fit_test_data(self, df=None):
        """
        Uses the fitted TFIDF vectorizer to create new TFIDF vectors
        for unseen data
        """
        if not df:
            df = self.test_corpus
        if not df:
            print("Please provide a DataFrame with test data")

        vector_df = df.apply(self.fit_new_sentence)
        matrix = vector_df.to_numpy()
        return matrix

    def fit_new_sentence(self, words):
        """
        Uses the fitted TFIDF vectorizer to create new TFIDF vectors
        on a sentence in str or list shape
        """
        if type(words) == str:
            words = words.lower().strip().split()
        n = len(words)
        vector = np.zeros(self.N_W)
        counter = collections.Counter(words)
        for word in words:
            try:
                word_id = self.word_to_index(word)
                TF = counter[word] / n
                IDF = (math.log((1 + self.N_D) / (self.doc_freq[word] + 1))) + 1
                TF_IDF = TF * IDF
                vector[word_id] = TF_IDF
            except KeyError: # if the word is not in the vocabulary
                pass
        vector = self.normalize(vector, axis=0)
        return vector

    def corpus_as_id(self):
        """
        get the corpus as lists of word ids, instead of words
        """
        id_corpus = self.corpus.apply(lambda l: [self.word_to_index(word) for word in l])
        id_corpus = id_corpus.values.tolist()
        return id_corpus

    def word_to_index(self, word: str):
        """ Returns the index corresponding to a word. """
        return self.vocab_w_i[word]

    def index_to_word(self, index: int):
        """ Returns the word corresponding to an index. """
        return self.vocab_i_w[index]

    def frequencies(self):
        """
        Calculates TF and DF values for all words in unique words
        """

        TF_list = []
        pbar = tqdm.tqdm(total=self.N_D)
        for doc in self.corpus:
            doc_dict = {}
            d_size = len(doc)
            tf_counts = collections.Counter(doc)
            for word in set(doc):
                try:
                    self.doc_freq[word] += 1
                except KeyError:
                    self.doc_freq[word] = 1
                doc_dict[word] = tf_counts[word] / d_size   # TF formula
            TF_list.append(doc_dict)
            pbar.update(1)
        return TF_list

    def filter_tokens(self):
        ## If df/self.N_D > self.DF_THRESHOLD- remove token
        self.doc_freq = {k:v for k,v in self.doc_freq.items() if v/self.N_D < self.DF_THRESHOLD}
        
    def TF_IFD(self):
        """
        Calculates TF-IDF values for all words in each doc
        """
        tf_idf_matrix = np.zeros((self.N_D, self.N_W))
        for doc_id, doc in enumerate(self.corpus):
            for word in doc:
                if word not in self.doc_freq:
                    continue
                word_id = self.word_to_index(word)
                idf = (math.log((1 + self.N_D) / (self.doc_freq[word] + 1))) + 1 # IDF formula with smoothing
                # idf = (math.log((self.N_D) / (self.doc_freq[word] + 1))) # IDF formula with smoothing
                tf = self.TF_list[doc_id][word]
                tf_idf = tf * idf
                tf_idf_matrix[doc_id][word_id] = tf_idf

        return self.normalize(tf_idf_matrix)

    def normalize(self, matrix, order=2, axis=1):
        """
        Performs L2 normalization
        """
        l2 = np.atleast_1d(np.linalg.norm(matrix, order, axis))
        l2[l2==0] = 1
        return matrix / np.expand_dims(l2, axis)


if __name__ == "__main__":
    import time
    data_path = "abcnews_lem_stem.pickle"
    dl = DataLoader()
    n_rows = 10000
    corpus = dl.load(data_path, n_rows=n_rows)

    start = time.time()
    TFIDF = TFIDFVectorizer()

    TFIDF_matrix = TFIDF.fit(corpus)
    print("Total time:", time.time() - start)
