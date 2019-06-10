import argparse
# Necessary for virtual environment
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn.cluster as cluster
from sklearn.decomposition.pca import PCA
import time


class KMeansClusterManager():
    """
    Template for building, training and predicting with a neural network
    """

    def __init__(self, random_seed):
        """
        Expose internal objects
        """
        np.random.seed(int(random_seed))

        self._df = None
        self._norm_df = None

        self._cluster_centers = None
        self._cluster_labels = None
        self._clusters = None
        self._sse = None

    def load_data(self):
        """
        Load mock data
        """
        start_time = time.time()

        from sklearn.datasets import load_iris
        iris_data = load_iris()
        self._df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
        species_names = list(iris_data.target_names)
        self._df['species'] = [species_names[i] for i in list(iris_data.target)]
        self._norm_df = self._normalize_df(self._df, list(range(self._df.shape[1] - 1)))

        # print(self._df.shape)
        # print(self._df.head())
        # print(self._norm_df.shape)
        # print(self._norm_df.head())

        print('### LOADING THE DATA {0} TOOK SECONDS ###'.format(round(time.time() - start_time, 1)))

    def _normalize_df(self, raw_df, col_index):
        """
        Normalizes the columns of a dataframe
        """
        result_df = raw_df.copy()
        columns = list(raw_df.columns)
        result_df.columns = columns
        for i in col_index:
            result_df.iloc[:, i] = (result_df.iloc[:, i] - np.mean(raw_df.iloc[:, i])) / np.std(raw_df.iloc[:, i])
        return(result_df)

    def _unnormalize_df(self, raw_df, norm_df, col_index):
        """
        Un-normalizes the columns of a dataframe according to the associated dataframe
        """
        result_df = norm_df.copy()
        result_df.columns = list(raw_df.columns)
        for i in col_index:
            result_df.iloc[:, i] = (result_df.iloc[:, i] * np.std(raw_df.iloc[:, i])) + np.mean(raw_df.iloc[:, i])
        return(result_df)

    def run_clustering(self, cluster_number):
        """
        Run the k-means clustering algorithm on the input data
        """
        start_time = time.time()

        sub_df = self._norm_df.iloc[:, :(self._df.shape[1] - 1)]
        kmeans = cluster.KMeans(n_clusters=cluster_number).fit(sub_df)
        self._cluster_centers = [list(center) for center in kmeans.cluster_centers_]
        self._cluster_labels = list(kmeans.labels_)
        self._clusters = [[list(row) for index, row in sub_df.iterrows() if self._cluster_labels[index] == cluster_number]
                          for cluster_number in range(cluster_number)]
        self._sse = sum([sum([self._compute_distance(self._clusters[cluster_number][i], self._cluster_centers[cluster_number])
                              for i in range(len(self._clusters[cluster_number]))])
                         for cluster_number in range(cluster_number)])

        # print(len(self._cluster_centers))
        # print(self._cluster_centers[:5])
        # print(len(self._cluster_labels))
        # print(self._cluster_labels[:5])
        # for i in range(len(self._clusters)):
        #     print(len(self._clusters[i]))
        #     print(self._clusters[i][:5])
        # print(self._sse)

        print('### CLUSTERING THE DATA WITH {0} CLUSTERS {1} SECONDS ###'.format(cluster_number, round(time.time() - start_time, 1)))

    def _compute_distance(self, vec_1, vec_2):
        """
        Computes the Euclidean distance between two given vectors
        """
        if len(vec_1) != len(vec_2):
            return(None)
        ans = 0
        for i in range(len(vec_1)):
            ans += pow(vec_1[i] - vec_2[i], 2)
        return(np.sqrt(ans))

    def plot_clusters(self, visualization_plot_path):
        """
        Plot the clustering results according to a pca decomposition
        """
        pca = PCA(n_components=2)
        pca.fit(self._norm_df.iloc[:, :(self._df.shape[1] - 1)])
        pca_transform = [list(row) for row in list(pca.fit_transform(self._norm_df.iloc[:, :(self._df.shape[1] - 1)]))]
        pca_clusters = [[pca_transform[i] for i in range(len(self._cluster_labels)) if self._cluster_labels[i] == cluster_number]
                        for cluster_number in range(len(self._cluster_centers))]

        # print(pca.components_)
        # print(pca.explained_variance_ratio_)
        # print(pca_transform[:5])
        # for i in range(len(pca_clusters)):
        #     print(len(pca_clusters[i]))
        #     print(pca_clusters[i][:5])

        plt.rcParams["figure.figsize"] = (16, 12)

        BOUNDS = (-3, 3)
        plt.xlim(BOUNDS[0], BOUNDS[1])
        plt.ylim(BOUNDS[0], BOUNDS[1])

        COLORS = ['red', 'green', 'blue', 'yellow', 'orange']
        for i in range(len(COLORS)):
            plt.scatter([row[0] for row in pca_clusters[i]], [row[1] for row in pca_clusters[i]], color=COLORS[i], alpha=0.8)

        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('K-Means Clustering PCA')

        plt.savefig(visualization_plot_path)
        plt.clf()

    def plot_elbow_graph(self, cluster_sizes, elbow_plot_path):
        """
        Plot the SSE elbow graph
        """
        sses = []
        for cluster_size in cluster_sizes:
            self.run_clustering(cluster_size)
            sses.append(float(self._sse))

        plt.rcParams["figure.figsize"] = (16, 12)

        plt.xlim(min(cluster_sizes) - 0.1, max(cluster_sizes) + 0.1)
        plt.ylim(min(sses) - 0.1, max(sses) + 0.1)

        plt.plot(cluster_sizes, sses, color='blue', alpha=0.8)

        plt.xlabel('Number of Clusters')
        plt.ylabel('SSE')
        plt.title('K-Means Elbow Plot')

        plt.savefig(elbow_plot_path)
        plt.clf()


def run(output_dir):
    """
    Run the program using the cli inputs
    """
    RANDOM_SEED = 0

    OUTPUT_DIR = str(output_dir)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    VISUALIZATION_PLOT_PATH = os.path.join(OUTPUT_DIR, 'k_means_plot.png')
    # ELBOW_PLOT_PATH = os.path.join(OUTPUT_DIR, 'k_means_elbow.png')

    cluster_manager = KMeansClusterManager(RANDOM_SEED)
    cluster_manager.load_data()
    cluster_manager.run_clustering(5)
    cluster_manager.plot_clusters(VISUALIZATION_PLOT_PATH)
    # cluster_manager.plot_elbow_graph(list(range(1, 10+1)), ELBOW_PLOT_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='K-Means Clustering Example')
    parser.add_argument('output_dir', help='Directory where the results are outputted')
    args = parser.parse_args()

    try:
        run(args.output_dir)
    except Exception as e:
        print(e)
