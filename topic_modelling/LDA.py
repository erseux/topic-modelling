from topic_modelling.data_loader import DataLoader
import numpy as np
import time
import pandas as pd
import math
from wordcloud import WordCloud
import matplotlib.pyplot as plt

class LDA:
    def __init__(self, corpus: pd.DataFrame, alpha=0.1, beta=0.1):
        """
        Initializes a LDA model
        input:
            corpus: a pandas DataFrame corresponding to the output from DataLoader.load()
            alpha: the alpha prior
            beta: the beta prior
        """
        self.ALPHA = alpha    # the Dirichlet prior for the document topic distribution
        self.BETA = beta   # the Dirichlet prior for the topic word distribution
        self.ITERATIONS = None
        self.MAX_WORDS_IN_TOPIC = None    # the N best words to examine from every topic

        self.corpus = corpus
        self.unique_words = self.get_unique()
        self.vocab_i_w = list(sorted(self.unique_words))
        self.vocab_w_i = {j:i for i,j in enumerate(self.vocab_i_w)}

        self.N_T = None    # number of topics
        self.N_D = len(self.corpus)   # number of documents
        self.N_W = len(self.unique_words) # size of the vocabulary

        self.docs_as_id = [[self.vocab_w_i[word] for word in doc] for doc in self.corpus]

    def setup(self):
        """
        Initializes the matrixes and vectors for the Gibbs sampling
        """
        self.doc_word_matrix = []
        self.doc_topic_matrix = np.zeros((self.N_D, self.N_T)) + self.ALPHA
        self.topic_word_matrix = np.zeros((self.N_T, self.N_W)) + self.BETA
        self.topic_vector = np.zeros(self.N_T) + self.N_W * self.BETA
        self.random_init()
    
    def random_init(self):
        """
        For each document d, for each word w in document:
            sample a topic z from the conditional probability for [d][w]
            assign doc_word_matrix[d][w] to z
        Increment self.doc_topic_matrix, self.topic_word_matrix and self.topic_vector
        for each seen topic z.
        """
        for d, doc in enumerate(self.docs_as_id):
            topics_for_doc = []
            for w in doc:
                pz = np.divide(np.multiply(self.doc_topic_matrix[d, :], self.topic_word_matrix[:, w]), self.topic_vector)
                z = np.random.multinomial(1, pz / pz.sum()).argmax() # sample a topic z
                topics_for_doc.append(z)
                self.doc_topic_matrix[d, z] += 1
                self.topic_word_matrix[z, w] += 1
                self.topic_vector[z] += 1
            self.doc_word_matrix.append(topics_for_doc)

    def get_unique(self):
        """
        Gets unique words from corpus.
        """
        unique = set()
        for row in self.corpus:
            for word in row:
                if word not in unique:
                    unique.add(word)
        return unique

    def gibbs_sampling(self):
        # resample topics for each document and each word
        for d, doc in enumerate(self.docs_as_id):
            for index, w in enumerate(doc):
                z = self.doc_word_matrix[d][index] # fetch current topic

                # subtract 1 from the original topic related amount for the current word in the current document
                self.doc_topic_matrix[d, z] -= 1
                self.topic_word_matrix[z, w] -= 1
                self.topic_vector[z] -= 1

                # re calculate the topic probabilities for the current word in the current document
                pz = np.divide(np.multiply(self.doc_topic_matrix[d, :], self.topic_word_matrix[:, w]), self.topic_vector)
                
                # sample from the current distributions
                pz /= pz.sum() # normalize the probability vector to make sure they sum to 1
                z = np.random.multinomial(1, pz).argmax() # sample a topic from the multinomial dist given by pz

                self.doc_word_matrix[d][index] = z

                # add 1 to the topic that was just sampled for the current word, for the current doc
                self.doc_topic_matrix[d, z] += 1
                self.topic_word_matrix[z, w] += 1
                self.topic_vector[z] += 1

    def perplexity(self):
        """
        Calculate the topic perplexities
        """
        nd = np.sum(self.doc_topic_matrix, 1)
        n = 0
        likelihood = 0.0
        for d, doc in enumerate(self.docs_as_id):
            for w in doc:
                likelihood = likelihood + np.log(
                    (
                        (self.topic_word_matrix[:, w] / self.topic_vector) * 
                        (self.doc_topic_matrix[d, :] / nd[d])   
                    ).sum()
                )
                n = n + 1
        return np.exp(likelihood/(-n))

    def get_topics(self, max_words_per_topic=20):
        """
        Get the n best words for each topic
        input:
            max_words_per_topic: int
        return:
            topicwords - List[List[str]] for all topics, for all best words per topic
            weights - the weights for each (of the best) word in each topic
        """
        self.MAX_WORDS_IN_TOPIC = max_words_per_topic
        topicwords = []
        weights = []
        for z in range(0, self.N_T):
            best_ids = self.topic_word_matrix[z, :].argsort()
            topicword = []
            for j in best_ids:
                topicword.insert(0, self.vocab_i_w[j])
            topicwords.append(topicword[0 : min(self.MAX_WORDS_IN_TOPIC, len(topicword))])
            weights.append(self.topic_word_matrix[z, best_ids][0 : min(self.MAX_WORDS_IN_TOPIC, len(topicword))])
        return topicwords, weights

    def fit(self, num_topics=10, iterations=50, verbosity=True):
        """
        Fit/train the LDA model
        input:
            num_topics: int - the n topics
            iterations: int - n iterations to run the sampling
            verbosity: bool - if True, print all iterations and iteration losses, else quiet
        """
        self.N_T = num_topics
        self.ITERATIONS = iterations

        self.setup()

        perplexities = []
        for i in range(0, self.ITERATIONS):
            self.gibbs_sampling()
            perplexity = self.perplexity()
            perplexities.append((i, perplexity))
            if verbosity:
                print(f"{time.strftime('%X')} Iteration: {i} Perplexity: {perplexity}")
        
        self.perplexities = perplexities

    def plot_perplexity(self, path="images/LDA/perplexity"):
        """
        Plots the perplexities for each iteration from the sampling
        """
        plt.plot(*zip(*self.perplexities))
        plt.title(f"Perplexity for {self.N_D} documents and {self.N_T} topics")
        plt.savefig(f"{path}/{self.N_D}rows_{self.N_T}topics_{self.ITERATIONS}it.png")

    def plot_word_clouds_all(self, path="images/LDA/wordcloud"):
        """
        Plot wordclouds for all topics in the model
        """
        clusters, weights = self.get_topics()
        topics = self.N_T 
        x_dim = math.ceil(topics / 3)

        fig, axs = plt.subplots(3, x_dim, figsize=(24, 14))
        for n in range(len(clusters)):
            j, i = divmod(n, 3)
            plot_word_cloud(clusters[n], weights[n], axs[i, j], n)
        axs[-1, -1].axis('off')
        
        fig.suptitle(f"Wordclouds n={self.N_D}", fontsize=30)
        plt.savefig(f"{path}/cloud_{self.N_D}_{self.N_T}.png")

def plot_word_cloud(cluster: list, weights: list, ax, n):
    """
    Generates a single subplot wordcloud
    """
    freqs = {word: weight * 10 for word, weight in zip(cluster, weights)}
    wc = WordCloud(background_color="white", width=800, height=500)
    wc = wc.generate_from_frequencies(freqs)
    ax.set_title('Topic %d' % (n + 1))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")

if __name__ == "__main__":
    data = DataLoader()
    data_path = "data/abcnews_lem_stem.pickle"
    dl = DataLoader()
    corpus = dl.load(data_path, n_rows=5000)

    print("prepared data for LDA")

    lda = LDA(corpus)
    print("initialized LDA")

    lda.fit(verbosity=True, num_topics=9, iterations=50)
    lda.plot_word_clouds_all()
    # lda.plot_perplexity()
    topics, _ = lda.get_topics()
    for topic in topics:
        print(topic)