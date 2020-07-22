import numpy as np
import pandas as pd
import dgl
import os
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import networkx as nx
import torch
import glob
import csv
from utils import graph_utils, style
# from .nx_utils import get_graph_from_data
import pickle
import logging
from sklearn.model_selection import KFold

class Dataset(object):
    def __init__(self, data_path, directed, k):
        self.data_path = data_path
        self.node_features_path = os.path.join(data_path, 'features.csv')
        # self.node_features_dir = os.path.join(data_path, 'features.txt')
        # self.node_features_files = glob.glob(os.path.join(self.node_features_dir, '*'))
        self.edges_dir = os.path.join(data_path, 'network.csv')
        self.directed = directed
        self.label_path = os.path.join(data_path, 'processed_labels.csv')
        self.vertex_map_path = os.path.join(data_path, 'node_id_map.csv' )
        self.train_val_test_mask = os.path.join(data_path, 'mask.csv')
        self.dgl_pickle_path = os.path.join(data_path, 'dgl_graph.pkl')
        self.k = k
        self.load()
    
    def load_graph(self):
        # Graph and Edge Features
        logging.info('Reading dgl graph directly from {}'.format(self.dgl_pickle_path))
        with open(self.dgl_pickle_path, 'rb') as f:
            self.g= pickle.load(f)
        self.number_of_nodes = self.g.number_of_nodes()
        self.num_edge_feats = self.g.edata['edge_features'].shape[1]
        logging.info('dgl graph loaded successfully from {}'.format(self.dgl_pickle_path))
   
    def load_labels(self):
        # Ground Truth Labels
        logging.info('Loading Labels...')
        self.labels = pd.read_csv(self.label_path, 
                            delimiter=',', 
                            )
    
    def preprocess_labels(self):
        logging.info('filtering unused nodes in the label')
        # convert label to one-hot format
        logging.info('convert label to one-hot format')
        self.labels = self.labels.set_index('nodeId')
        self.labels.index = self.labels.index.astype(int)
        self.labels = self.labels.loc[~self.labels.index.duplicated(keep='last')]
        one_hot_labels = pd.get_dummies(data=self.labels, dummy_na=True, columns=['label']) # N X (#edge attr)  # one hot 
        one_hot_labels = one_hot_labels.drop(['label_nan'], axis=1)
        logging.info('Train, validation, test split')
        # train, val, test split
        logging.info('The mask file: {} doest not exist. Performing train, val, test split'.format(self.train_val_test_mask))
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        train_index, val_index = list(kf.split(self.labels.index))[self.k]
        train_id = self.labels.iloc[train_index].index
        val_id = self.labels.iloc[val_index].index

        self.train_id = train_id
        self.val_id = val_id
        self.test_id = val_id
            
        self.train_mask = np.zeros((self.number_of_nodes,)).astype(int)
        self.val_mask = np.zeros((self.number_of_nodes,)).astype(int)
        self.test_mask = np.zeros((self.number_of_nodes,)).astype(int)

        # train_ratio = 0.8
        np.random.seed(1)
        self.train_mask[list(train_id)] = 1
        self.val_mask[list(val_id)] = 1
        # self.test_mask[list(test_id)] = 1
        self.test_mask[list(val_id)] = 1

        y = np.zeros(self.number_of_nodes)
        y[one_hot_labels.index] = np.argmax(one_hot_labels.values, 1)

        logging.info('train_mask shape: {}'.format(self.train_mask.shape))
        logging.info('val_mask shape: {}'.format(self.val_mask.shape))
        logging.info('test_mask shape: {}'.format(self.test_mask.shape))

    def load(self):
        logging.info('loading data...')
        # Load Graph and Edge features
        # Labels
        self.load_labels()
        # Load Graph
        self.load_graph()
        # Map vertex id to consecutive integers
        # self.load_vertex_id_map()
        # Preprocess ground truth label
        self.preprocess_labels()

        # self.num_classes = len(np.unique(self.labels))
        self.num_classes = len(np.unique(self.labels['label'].values))
