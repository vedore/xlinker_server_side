from sklearn.cluster import AgglomerativeClustering, KMeans, Birch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from src.joao_project.src.machine_learning.clustering import Clustering
from src.joao_project.src.machine_learning.regression import Regression


class AgglomerativeClusteringCPU(Clustering):
    
    @classmethod
    def train(cls, embeddings):
        defaults = {
            'n_clusters': 16,   
            'memory': ''        
        }

        # defaults.update(kwargs)
        model = AgglomerativeClustering(**defaults)
        model.fit(embeddings)
        return cls(model=model, model_type='HierarchicalCPU')

    def get_labels(self):
        return self.model.labels_
        
class LogisticRegressionCPU(Regression):

    @classmethod
    def train(cls, X_train, y_train):
        defaults = {
            'random_state': 0,
            'solver': 'lbfgs',
            'max_iter': 100,
            'verbose': 1
        }

        # SVM
        model = LogisticRegression(**defaults)
        # X_train = csr_matrix(X_train)
        model.fit(X_train, y_train)
        return cls(model=model, model_type='LogisticRegressionCPU')
    
class KMeansCPU(Clustering):

    @classmethod
    def train(cls, X_train):
        defaults = {
            'n_clusters': 16,
            'max_iter': 20,
            'random_state': 0,
            'n_init': 10
        }

        print("Normalize")
        X_normalized = normalize(X_train)
        print("Running Model")
        model = KMeans(**defaults)
        model.fit(X_normalized)
        return cls(model=model, model_type='KMeansCPU')
    
    def get_labels(self):
        return self.model.labels_
    
class BirchCPU(Clustering):

    @classmethod
    def train(cls, X_train):
        defaults = {
            'threshold': 1,
            'branching_factor': 50,
            'n_clusters': 16,
            'compute_labels': True,
        }
        # X_normalized = normalize(X_train)
        model = Birch(**defaults)
        model.fit(X_train)
        return cls(model=model, model_type='BirchCPU')
    
    def get_labels(self):
        return self.model.labels_