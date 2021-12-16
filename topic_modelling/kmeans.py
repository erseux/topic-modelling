from sklearn.cluster import KMeans
from topic_modelling.TF_IDF import TFIDFVectorizer
from topic_modelling.data_loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import math


class KMeansCluster:
    def __init__(self, corpus, TFIDF_method="native", num_clusters=5):
        self.corpus = corpus
        self.words = None
        self.vectorizer = None
        self.index_to_word = None

        self.method = TFIDF_method
        self.N_docs = len(corpus)
        self.cluster_centers = None

        # initalize TF_IDF model
        if TFIDF_method == "native":
            self.matrix = self.TFIDF_native(corpus)
        elif TFIDF_method == "sklearn":
            self.matrix = self.TFIDF_sklearn(corpus)

        # parameters
        self.NUM_CLUSTERS = num_clusters
        self.clustered_documents = None

    def TFIDF_native(self, corpus):
        """
        Uses our TF-IDF vectorizer
        return:
            TF-IDF matrix
        """
        TF_IDF = TFIDFVectorizer()
        self.vectorizer = TF_IDF
        matrix = TF_IDF.fit(corpus, max_df=0.05)

        self.index_to_word = TF_IDF.vocab_i_w
        return matrix

    def TFIDF_sklearn(self, corpus):
        """
        Uses sklearns TFIDF vectorizer (which is faster)
        """
        corpus = corpus.apply(lambda l: " ".join(l))
        vectorizer = TfidfVectorizer()
        self.vectorizer = vectorizer
        matrix = vectorizer.fit_transform(corpus)
        self.index_to_word = vectorizer.get_feature_names_out()
        return matrix

    def cluster_documents(self, verbosity=True):
        """
        clusters documents into self.NUM_CLUSTERS clusters
        input:
            verbosity: bool - if True, print all clusters and the
                docs they contain, else quiet
        return:
            clustered_documents: List[List[str]] for each cluster, for each
                document in cluster
            cluster_assignment: np.Array - array of the cluster assignments
                for every doc
        """
        clustering_model = KMeans(n_clusters=self.NUM_CLUSTERS, random_state=3425)
        clustering_model.fit(self.matrix)
        cluster_assignment = clustering_model.labels_

        clustered_documents = [[] for i in range(self.NUM_CLUSTERS)]
        for doc_id, cluster_id in enumerate(cluster_assignment):
            clustered_documents[cluster_id].append(self.corpus[doc_id])

        if verbosity:
            for n, cluster in enumerate(clustered_documents):
                print("Cluster ", n + 1)
                print(*cluster, sep="\n")
                print("")

        # save these for later
        self.clustered_documents = clustered_documents
        self.cluster_centers = clustering_model.cluster_centers_ 

        return clustered_documents, cluster_assignment
        
    def get_best_words(self, n_words=10):
        """Gets the n best words from each cluster, and their centroid vector score"""
        for cluster in self.cluster_centers:
            idx = np.argpartition(cluster, -n_words)[-n_words:]
            indices = idx[np.argsort((-cluster)[idx])]
            words = [self.index_to_word[i] + f" ({cluster[i]:.2f})" for i in indices]
            print(words)

    def elbow_test(self, min_c=5, max_c=14, show=False, save=True):
        """Performs and plots an elbow test"""
        wcss = []
        for n in range(min_c,max_c):
            print(f"fitting with {n} clusters")
            clustering_model = KMeans(n_clusters=n,init='k-means++',max_iter=300,n_init=10,random_state=0)
            clustering_model.fit(self.matrix)
            wcss.append(clustering_model.inertia_)
            
        plt.plot(range(min_c,max_c),wcss)
        plt.title(f'The Elbow Method ({self.method} TF-IDF), n={self.N_docs}')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        if save:
            plt.savefig(f'images/kmeans/elbow/elbow_{self.method}_{self.N_docs}.png')
        if show:
            plt.show()

    def plot_word_clouds_all(self):
        """Plots wordclouds for the clusters"""
        clusters = self.clustered_documents
        topics = self.NUM_CLUSTERS 
        x_dim = math.ceil(topics / 3)

        fig, axs = plt.subplots(3, x_dim, figsize=(24, 14))
        for n in range(len(clusters)):
            j, i = divmod(n, 3)
            plot_word_cloud(clusters[n], axs[i, j], n)
        axs[-1, -1].axis('off')
        
        fig.suptitle(f"Wordclouds n={self.N_docs}", fontsize=30)
        plt.savefig(f"images/kmeans/wordcloud/cloud_{self.N_docs}_{self.NUM_CLUSTERS}.png")

def get_word_counts(cluster: list):
    """Gets the most common words from a topic"""
    MAX_WORDS_PER_TOPIC = 20
    words = [word for document in cluster for word in document]
    counts = collections.Counter(words)
    best = counts.most_common(MAX_WORDS_PER_TOPIC)
    counts = {k:v for k,v in best}
    return counts

def plot_word_cloud(cluster: list, ax, n): 
    freqs = get_word_counts(cluster)
    wc = WordCloud(background_color="white", width=800, height=500)
    wc = wc.generate_from_frequencies(freqs)
    ax.set_title('Topic %d' % (n + 1))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")


if __name__ == "__main__":
    import time
    data_path = "data/abcnews_lem_stem.pickle"

    dl = DataLoader()
    corpus = dl.load(data_path, n_rows=5000)
    start = time.time()
    cluster = KMeansCluster(corpus, TFIDF_method="sklearn", num_clusters=9)
    # cluster.elbow_test()

    doc_clusters, _ = cluster.cluster_documents(verbosity=False)
    t = time.time() - start
    
    cluster.get_best_words()
    cluster.plot_word_clouds_all()

