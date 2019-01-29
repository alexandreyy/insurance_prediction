'''
Created on 02/12/2015

@author: Alexandre Yukio Yamashita
'''
from data.data import Data
from sklearn.decomposition import PCA


if __name__ == '__main__':
    path = "../homesite_data/resources/parsed_data.bin"
    homesite = Data()
    homesite.load_parsed_data(path)
    print homesite.train_x.shape
    pca = PCA(n_components = 0.99)
    pca.fit(homesite.train_x)
    print pca.transform(homesite.train_x).shape