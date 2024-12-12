import cv2
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import pickle
import logging

logging.info("Computing codebook with MiniBatchKMeans")
orb_descriptors = np.load("orb_descriptors.npy")
#codebook = MiniBatchKMeans(n_clusters=128,batch_size= 1000, init='k-means++', n_init=10, verbose=1).fit(orb_descriptors)
codebook = KMeans(n_clusters=128, init='k-means++', n_init=10, verbose=1).fit(orb_descriptors)
with open("codebook.pkl", "wb") as f:
    pickle.dump(codebook, f)
logging.info("Saved codebook to 'codebook.pkl'")