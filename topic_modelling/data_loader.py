from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk import word_tokenize
from pandas.core.frame import DataFrame
import pandas as pd
import re


class DataLoader:
    """
    Load and process data in csv or DataFrame form
    Save processed data to pickle, and reload same pickle file for faster data access.
    """
    def __init__(self, language="english", stemming=True, lemmatize=True):

        self.__path = None

        self.__TEXT_COLUMN = "headline_text"

        self.__stemming = stemming
        self.__lemmatize = lemmatize

        self.__stopwords = set(stopwords.words(language))

        self.__lemmatizer = WordNetLemmatizer()

        self.__stemmer = SnowballStemmer(language=language)
        
        self.__raw_data = None
        
        self.__processed_data = None

    def process_data(self, text, remove_special_chars=False):
        """
        Input: text = line for line in raw_data
        1. Tokenize
        2. Remove stopwords
        3. Lemmatize and Stem
        Return: processed data
        """
        pattern = re.compile(r'[^a-z]+')
        text = text.lower()
        text = pattern.sub(" ", text).strip()
        words = self.tokenize(text)
        
        processed_words = []
        for word in words:
            if word not in self.__stopwords:
                processed_word = self.lemmatize_stem(word)
                if len(processed_word) > 2:
                    processed_words.append(processed_word)
        return processed_words
        
    def read_data(self):
        """
        Reads csv file
        Return: self.__raw_data
        """
        data = pd.read_csv(self.__path).drop_duplicates(self.__TEXT_COLUMN)
        return data[self.__TEXT_COLUMN]

    def tokenize(self, text):
        """
        Input: text
        Return: tokenized text
        """
        return word_tokenize(text)

    def lemmatize_stem(self, word):
        """
        Input: word
        Return: Lemmatized and stemmed word
        """
        o_word = word
        if self.__lemmatize:
            word = self.__lemmatizer.lemmatize(word)
        if self.__stemming:
            word = self.__stemmer.stem(word)
        if word in self.__stopwords:
            word = ""
        return word
        
    def load(self, path: str, n_rows=None, text_column=None):
        """
        Input:
        1. Path of existing processed data (.pickle) or path of raw data (.csv)
        2. Optional: number of rows (n_rows) to process in data
        """
        self.__path = path
        if text_column:
            self.__TEXT_COLUMN = text_column
        if path[-6:] == "pickle":
            self.__processed_data = pd.read_pickle(path)
            if n_rows:
                self.__processed_data = self.__processed_data.head(n_rows)
            print("Read processed data from: {}".format(path))
        elif path[-3:] == "csv":
            self.__raw_data = self.read_data()
            if n_rows:
                self.__raw_data = self.__raw_data.head(n_rows)
            self.__processed_data = self.__raw_data.apply(self.process_data)
            self.__processed_data = self.__processed_data.reset_index(drop=True)
            print("Read processed data from: {}".format(path))
        
        else:
            raise FileNotFoundError("Unexpected path format")
            
        return self.__processed_data

    def load_df(self, data_df: DataFrame, n_rows=None):
        """
        Processes data from a DataFrame with a "data" column containing text
        """
        self.__raw_data = data_df.drop_duplicates("data")

        if n_rows:
            self.__raw_data = self.__raw_data.head(n_rows)

        self.__raw_data["data"] = self.__raw_data["data"].apply(self.process_data)
        self.__processed_data = self.__raw_data
        self.__processed_data = self.__processed_data.reset_index(drop=True)

        print("Processed data")
        return self.__processed_data

    def save(self, path):
        """
        Input: path for output file
        Saves processed data to path.pickle
        """
        self.__processed_data.to_pickle(path)
        print("Saved processed data to: {}".format(path))

if __name__ == "__main__":
    import time
    csv_path = "data/abcnews-date-text.csv"
    pickle_path = "data/abcnews_lem_stem.pickle"

    data = DataLoader(lemmatize=True, stemming=True)

    t1 = time.time()
    _ = data.load(pickle_path)
    t2 = time.time()
    t = t2 - t1
    data.save(pickle_path)
    print(f"loading data took {t} seconds")