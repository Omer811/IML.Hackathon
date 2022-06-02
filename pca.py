from data_loader import Loader
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler

X_PATH = "Mission2_Breast_Cancer/train.feats.csv.pickled"

if __name__ == '__main__':
    loader = Loader(pickled_path=X_PATH)
    loader.load_pickled()
    data = loader.get_data()
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    pca = PCA(n_components=data.shape[1])
    pca.fit(data)
    print(np.round(np.cumsum(pca.explained_variance_ratio_),4))
    print(np.round(pca.components_[0], 2))