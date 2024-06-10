from numpy.linalg import inv, matrix_rank, pinv
import numpy as np
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import repeat
from sklearn.mixture import GaussianMixture
from scipy.stats import f
from tqdm import tqdm


class CDD():
    def __init__(self, clusters_data=np.nan, max_k=10, random=0, method='gmm', k=np.nan, c=1000000, n_examples_in_memory=0):
        self.max_k = max_k
        self.random = random
        self.method = method
        self.k = k
        self.c = c
        self.clusters_data = clusters_data
        self.n_examples_in_memory = n_examples_in_memory

    def fit(self, x_train):
        """
        Parameters
        ----------
        X_train : TYPE : array
            DESCRIPTION : Training data, consist only normal observations (one class).
        max_k : TYPE : int
            DESCRIPTION : Maximum clusters for the training set (to represent the normal class).
        random : TYPE : int, optional
            DESCRIPTION : Random seed. The default is 0.
        method : TYPE : string, optional
            DESCRIPTION : Clustering method for the training set, can be either kmeans or gmm. The default is 'kmeans'.

        Returns
        -------
        TYPE : list
            DESCRIPTION : Statistics of the clustering of the training data, each element in the list is list of :
                mean vector, covariance matrix and number of observations for each cluster.
                :param x_train:
        """
        if self.method == 'kmeans':

            def choose_k(data, max_k, random=0):
                # choosing number of clusters with silhouette coefficient
                scores = np.zeros(max_k - 1)
                for k in range(2, max_k + 1):
                    kmeans = KMeans(n_clusters=k, random_state=random)
                    preds = kmeans.fit_predict(data)
                    score = silhouette_score(data, preds, metric='euclidean')
                    scores[k - 2] = score
                    print(k)
                return np.argmax(scores) + 2
            if np.isnan(self.k):
                k = choose_k(x_train, self.max_k, self.random)
            else:
                k = self.k

            def get_clusters_data(data, k, random=0):
                # this function  for each cluster its center,covariance matrix and number of examples
                # assume data is scaled
                kmeans = KMeans(n_clusters=k, random_state=random).fit(data)
                labels = kmeans.labels_
                lst = [[] for i in repeat(None, k)]
                for c in range(k):
                    temp_data = data[labels == c, :]
                    mean = np.mean(temp_data, axis=0)
                    cov = np.cov(np.transpose(temp_data))
                    m = temp_data.shape[0]
                    lst[c].extend([mean, cov, m])
                return lst

            self.clusters_data = get_clusters_data(x_train, k)
        if self.method == 'gmm':
            def choose_k(data, max_k, random=0):
                scores = np.zeros(max_k - 1)
                for c in range(2, max_k + 1):
                    gmm = GaussianMixture(n_components=c, random_state=random).fit(data)
                    preds = gmm.predict(data)
                    score = silhouette_score(data, preds, metric='euclidean')
                    scores[c - 2] = score
                return np.argmax(scores) + 2
            if np.isnan(self.k):
                k = choose_k(x_train, self.max_k, self.random)
            else:
                k = self.k

            def get_clusters_data(data, k, random=0):
                # this function estimates for each cluster its center, covariance matrix and number of examples
                # assume data is scaled
                gmm = GaussianMixture(n_components=k, random_state=random)
                labels = gmm.fit_predict(data)
                labels_to_delete = []
                for label, count in enumerate(np.bincount(labels)):
                    if count == 1:
                        labels_to_delete.append(label)

                relevent_indices = [ind for ind, label in enumerate(labels) if label not in labels_to_delete]
                data = data[relevent_indices, :]
                labels = labels[relevent_indices]
                k = len(np.unique(labels))

                lst = [[] for i in repeat(None, k)]
                for ind, c in enumerate(np.unique(labels)):
                    temp_data = data[labels == c, :]
                    mean = np.mean(temp_data, axis=0)
                    cov = np.cov(np.transpose(temp_data))
                    m = temp_data.shape[0]
                    lst[ind].extend([mean, cov, m])
                return lst

            self.clusters_data = get_clusters_data(x_train, k)

    def predict(self, new_data, update=0.1, alpha=0.05):
        """
        Parameters
        ----------
        Clusters_data : TYPE : list
                DESCRIPTION : Statistics of the clustering of the training data, each element in the list is list of :
                mean vector, covariance matrix and number of observations for each cluster.
        new_data : TYPE : array
            DESCRIPTION : Test data (cosist both normal and novel classes).
        update : TYPE : float, optional
            DESCRIPTION : The fraction of new observations appended to the cluster,
            which requier updating its statistics. The default is 0.1.
        alpha : TYPE : float (between zero to one), optional
            DESCRIPTION : Significant level for the SPC test. The default is 0.05.

        Returns
        -------
        TYPE : array (vector)
            DESCRIPTION : predictions for the test data, each element in the vector
            equal to 1 if normal and -1 if novel.

        """
        # SPC test using Hotelling chart
        def hotelling(x, mean, s):
            P = mean.shape[0]
            a = x - mean
            ##check rank to decide the type of inverse(regular or adjused to non inverses matrix)
            if matrix_rank(s) == P:
                b = inv(np.array(s))
            else:
                b = pinv(np.array(s))
            c = np.transpose(x - mean)
            T2 = np.matmul(a, b)
            T2 = np.matmul(T2, c)
            return T2

        k = len(self.clusters_data)
        new_obs_each_cluster = [[] for x in range(k)]
        count_obs_each_cluster = np.zeros(k)
        M, P = new_data.shape
        anomalies = np.zeros(M)
        relevent_indices = [c for c in range(k) if self.clusters_data[c][2] > P]
        concepts = len(relevent_indices)

        def SPC(T2, M, N, alpha=0.05):
            F = scipy.stats.f.ppf(q=1 - alpha, dfn=N, dfd=M - N)
            UCL = (N * (M - 1) * (M + 1) * F) / (M ** 2 - M * N)
            return T2 > UCL

        def CDD_update(clusters_data, cluster_to_update, new_obs):
            new_obs = np.asarray(new_obs)
            new_mean = np.mean(new_obs, axis=0)
            new_cov = np.cov(np.transpose(new_obs))
            new_m = new_obs.shape[0]
            ls_to_update = clusters_data[cluster_to_update]
            p_new = new_m / (new_m + ls_to_update[2])
            ls_new = [[] for x in range(3)]
            ls_new[0] = new_mean * p_new + ls_to_update[0] * (1 - p_new)
            ls_new[1] = new_cov * p_new + ls_to_update[1] * (1 - p_new)
            ls_new[2] = new_m + ls_to_update[2]
            clusters_data[cluster_to_update] = ls_new
            return clusters_data

        def Hotelling_DistributionComparison(Xgag, Ygag, Xcov, Ycov, Nx, Ny, alpha=alpha):
            P = Xgag.shape[0]
            Pooled_cov = ((Nx - 1) * Xcov + (Ny - 1) * Ycov) / (Nx + Ny - 2)
            a = Xgag - Ygag
            if matrix_rank(Pooled_cov) == P:
                b = inv(np.array(Pooled_cov))
            else:
                try:
                    b = pinv(np.array(Pooled_cov))
                except:
                    return True
            c = np.transpose(Xgag - Ygag)
            T2 = np.matmul(a, b)
            T2 = np.matmul(T2, c)
            T2 = T2 * (Nx * Ny) / (Nx + Ny)
            F = T2 * (Nx + Ny - P - 1) / ((Nx + Ny - 2) * P)
            if Nx + Ny - 1 - P > 0:
                pval = 1 - scipy.stats.f.cdf(F, P, Nx + Ny - 1 - P)
                isnot_merged = pval < alpha
            else:
                isnot_merged = True
            return isnot_merged

        def is_cluster_merged(cluster, fitted, alpha):
            clusters = list(range(len(fitted)))
            clusters.remove(cluster)
            Xgag = fitted[cluster][0]
            Xcov = fitted[cluster][1]
            Nx = fitted[cluster][2]
            merged_cluster = np.nan
            for c in clusters:
                Ygag = fitted[c][0]
                Ycov = fitted[c][1]
                Ny = fitted[c][2]
                if not Hotelling_DistributionComparison(Xgag, Ygag, Xcov, Ycov, Nx, Ny, alpha):
                    merged_cluster = c
            return merged_cluster

        def merge_clusters(cluster1, cluster2, fitted):
            new_fitted = []
            for i, stat in enumerate(fitted):
                if i not in [cluster1, cluster2]:
                    new_fitted.append(stat)
            merged_size = fitted[cluster1][2] + fitted[cluster2][2]
            prop_first_cluster = fitted[cluster1][2] / merged_size
            merged_mean = prop_first_cluster * fitted[cluster1][0] + (1 - prop_first_cluster) * fitted[cluster2][0]
            merged_cov = prop_first_cluster * fitted[cluster1][1] + (1 - prop_first_cluster) * fitted[cluster2][1]
            new_fitted.append([merged_mean, merged_cov, merged_size])
            return new_fitted
        self.n_examples_in_memory = 0
        for m in tqdm(range(M)):
            obs = new_data[m, :]
            diffrent_from = np.zeros(k)
            T2 = np.zeros(k)
            for c in range(k):
                t2 = hotelling(obs, self.clusters_data[c][0], self.clusters_data[c][1])
                if c in relevent_indices:
                    spc = SPC(t2, self.clusters_data[c][2], P, alpha)
                    diffrent_from[c] = spc
                T2[c] = t2
            if sum(diffrent_from) != concepts:
                cluster = np.argmin(T2)
                new_obs_each_cluster[cluster].append(obs)
                count_obs_each_cluster[cluster] += 1
                self.n_examples_in_memory += 1
                if self.n_examples_in_memory >= self.c:
                    biggest_cluster = count_obs_each_cluster.index(max(count_obs_each_cluster))
                    self.n_examples_in_memory -= int(0.1*len(new_obs_each_cluster[biggest_cluster]))
                    new_obs_each_cluster[biggest_cluster] = \
                        new_obs_each_cluster[biggest_cluster][int(0.1*len(new_obs_each_cluster[biggest_cluster])):]
                anomalies[m] = 1
                if (count_obs_each_cluster[cluster] >= int(update * self.clusters_data[cluster][2])) & (
                        count_obs_each_cluster[cluster] > 1):
                    self.clusters_data = CDD_update(self.clusters_data, cluster, new_obs_each_cluster[cluster])
                    self.n_examples_in_memory -= count_obs_each_cluster[cluster]
                    count_obs_each_cluster[cluster] = 0
                    new_obs_each_cluster[cluster] = []
                    merged_cluster = is_cluster_merged(cluster, self.clusters_data, alpha)
                    if not np.isnan(merged_cluster):
                        self.clusters_data = merge_clusters(cluster, merged_cluster, self.clusters_data)
                        k -= 1
                    relevent_indices = [c for c in range(k) if self.clusters_data[c][2] > P]
                    concepts = len(relevent_indices)
            else:
                anomalies[m] = -1
        return anomalies

    def predict_proba(self, new_data, update=0.1, alpha=0.05):
        """
        Parameters
        ----------
        Clusters_data : TYPE : list
                DESCRIPTION : Statistics of the clustering of the training data, each element in the list is list of :
                mean vector, covariance matrix and number of observations for each cluster.
        new_data : TYPE : array
            DESCRIPTION : Test data (cosist both normal and novel classes).
        update : TYPE : float, optional
            DESCRIPTION : The fraction of new observations appended to the cluster,
            which requier updating its statistics. The default is 0.1.
        alpha : TYPE : float (between zero to one), optional
            DESCRIPTION : Significant level for the SPC test. The default is 0.05.

        Returns
        -------
        TYPE : array (vector)
            DESCRIPTION : predictions for the test data, each element in the vector is the probability 
            to be novel (anomaly score).

         """

        # SPC test using Hotelling chart
        def hotelling(x, mean, s):
            P = mean.shape[0]
            a = x - mean
            # check rank to decide the type of inverse(regular or adjused to non inverses matrix)
            if matrix_rank(s) == P:
                b = inv(np.array(s))
            else:
                b = pinv(np.array(s))
            c = np.transpose(x - mean)
            T2 = np.matmul(a, b)
            T2 = np.matmul(T2, c)
            return T2

        k = len(self.clusters_data)
        new_obs_each_cluster = [[] for x in range(k)]
        count_obs_each_cluster = np.zeros(k)
        M, P = new_data.shape
        anomalies = np.zeros(M)
        anomalie_scores = np.zeros(M)
        relevent_indices = [c for c in range(k) if self.clusters_data[c][2] > P]
        concepts = len(relevent_indices)

        def SPC(T2, M, N, alpha=0.05):
            F = scipy.stats.f.ppf(q=1 - alpha, dfn=N, dfd=M - N)
            UCL = (N * (M - 1) * (M + 1) * F) / (M ** 2 - M * N)
            return T2 > UCL

        def CDD_update(clusters_data, cluster_to_update, new_obs):
            new_obs = np.asarray(new_obs)
            new_mean = np.mean(new_obs, axis=0)
            new_cov = np.cov(np.transpose(new_obs))
            new_m = new_obs.shape[0]
            ls_to_update = clusters_data[cluster_to_update]
            p_new = new_m / (new_m + ls_to_update[2])
            ls_new = [[] for x in range(3)]
            ls_new[0] = new_mean * p_new + ls_to_update[0] * (1 - p_new)
            ls_new[1] = new_cov * p_new + ls_to_update[1] * (1 - p_new)
            ls_new[2] = new_m + ls_to_update[2]
            clusters_data[cluster_to_update] = ls_new
            return clusters_data

        def Hotelling_DistributionComparison(Xgag, Ygag, Xcov, Ycov, Nx, Ny, alpha=alpha):
            P = Xgag.shape[0]
            Pooled_cov = ((Nx - 1) * Xcov + (Ny - 1) * Ycov) / (Nx + Ny - 2)
            a = Xgag - Ygag
            if matrix_rank(Pooled_cov) == P:
                b = inv(np.array(Pooled_cov))
            else:
                b = pinv(np.array(Pooled_cov))
            c = np.transpose(Xgag - Ygag)
            T2 = np.matmul(a, b)
            T2 = np.matmul(T2, c)
            T2 = T2 * (Nx * Ny) / (Nx + Ny)
            F = T2 * (Nx + Ny - P - 1) / ((Nx + Ny - 2) * P)
            if Nx + Ny - 1 - P > 0:
                pval = 1 - scipy.stats.f.cdf(F, P, Nx + Ny - 1 - P)
                isnot_merged = pval < alpha
            else:
                isnot_merged = True
            return isnot_merged

        def is_cluster_merged(cluster, fitted, alpha):
            clusters = list(range(len(fitted)))
            clusters.remove(cluster)
            Xgag = fitted[cluster][0]
            Xcov = fitted[cluster][1]
            Nx = fitted[cluster][2]
            merged_cluster = np.nan
            for c in clusters:
                Ygag = fitted[c][0]
                Ycov = fitted[c][1]
                Ny = fitted[c][2]
                if not Hotelling_DistributionComparison(Xgag, Ygag, Xcov, Ycov, Nx, Ny, alpha):
                    merged_cluster = c
            return merged_cluster

        def merge_clusters(cluster1, cluster2, fitted):
            new_fitted = []
            for i, stat in enumerate(fitted):
                if i not in [cluster1, cluster2]:
                    new_fitted.append(stat)
            merged_size = fitted[cluster1][2] + fitted[cluster2][2]
            prop_first_cluster = fitted[cluster1][2] / merged_size
            merged_mean = prop_first_cluster * fitted[cluster1][0] + (1 - prop_first_cluster) * fitted[cluster2][0]
            merged_cov = prop_first_cluster * fitted[cluster1][1] + (1 - prop_first_cluster) * fitted[cluster2][1]
            new_fitted.append([merged_mean, merged_cov, merged_size])
            return new_fitted
        self.n_examples_in_memory = 0
        for m in tqdm(range(M)):
            obs = new_data[m, :]
            diffrent_from = np.zeros(k)
            T2 = np.zeros(k)
            for c in range(k):
                t2 = hotelling(obs, self.clusters_data[c][0], self.clusters_data[c][1])
                if c in relevent_indices:
                    spc = SPC(t2, self.clusters_data[c][2], P, alpha)
                    diffrent_from[c] = spc
                T2[c] = t2

            indices = np.argsort(T2)
            for ind in indices:
                if self.clusters_data[ind][2] > P:
                    tval = T2[ind]
                    cluster = ind
                    break

            cluster_size = self.clusters_data[cluster][2]
            Fstat = tval * ((cluster_size - P) * cluster_size) / (P * (cluster_size - 1) * (cluster_size + 1))
            anomalie_scores[m] = f.cdf(Fstat, P, cluster_size - P)

            if sum(diffrent_from) != concepts:
                cluster = np.argmin(T2)
                new_obs_each_cluster[cluster].append(obs)
                count_obs_each_cluster[cluster] = count_obs_each_cluster[cluster] + 1
                anomalies[m] = 1
                self.n_examples_in_memory += 1
                if self.n_examples_in_memory >= self.c:
                    biggest_cluster = count_obs_each_cluster.index(max(count_obs_each_cluster))
                    self.n_examples_in_memory -= int(0.1*len(new_obs_each_cluster[biggest_cluster]))
                    new_obs_each_cluster[biggest_cluster] = \
                        new_obs_each_cluster[biggest_cluster][int(0.1*len(new_obs_each_cluster[biggest_cluster])):]
                if (count_obs_each_cluster[cluster] >= int(update * self.clusters_data[cluster][2])) & (
                        count_obs_each_cluster[cluster] > 1):
                    self.clusters_data = CDD_update(self.clusters_data, cluster, new_obs_each_cluster[cluster])
                    self.n_examples_in_memory -= count_obs_each_cluster[cluster]
                    count_obs_each_cluster[cluster] = 0
                    new_obs_each_cluster[cluster] = []
                    merged_cluster = is_cluster_merged(cluster, self.clusters_data, alpha)
                    if not np.isnan(merged_cluster):
                        self.clusters_data = merge_clusters(cluster, merged_cluster, self.clusters_data)
                        k -= 1
                    relevent_indices = [c for c in range(k) if self.clusters_data[c][2] > P]
                    concepts = len(relevent_indices)
            else:
                anomalies[m] = -1
        return anomalie_scores
