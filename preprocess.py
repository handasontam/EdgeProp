import logging
import dgl
import numpy as np
import pandas as pd
import pickle
import csv
from utils import graph_utils, style
import os
import torch
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from dmt.utils import Params, set_logger, style




data_path = './data'
set_logger(os.path.join(data_path, 'preprocess.log'))
node_features_path = os.path.join(data_path, 'features.csv')
processed_node_features_path = os.path.join(data_path, 'processed_node_features.csv')
dynamic_edges_features_path = os.path.join(data_path, 'network.csv')
with open(dynamic_edges_features_path) as f:
    num_edge_feats = len(f.readline().strip().split(','))-2
directed = True
if directed:
    num_edge_feats = num_edge_feats * 2
label_path = os.path.join(data_path, 'labels.csv')
processed_label_path = os.path.join(data_path, 'processed_labels.csv')
vertex_map_path = os.path.join(data_path, 'node_id_map.csv' )
train_val_test_mask = os.path.join(data_path, 'mask.txt')
dgl_pickle_path = os.path.join(data_path, 'dgl_graph.pkl')


def load_node_features():
    # Node Features
    logging.info('Loading Node Features...')
    features = pd.read_csv(node_features_path, 
                           delimiter=',', 
                           )
    features = features.set_index('nodeId')
    return features

def hand_craft_edge_features():
    logging.info('mannually extract edge features...')
    transactions = pd.read_csv(dynamic_edges_features_path, 
                               delimiter=',')
    transactions = transactions.groupby(['srcId', 'dstId'])[['money', 'type']].agg(['mean', 'median', 'sum', 'count', 'std', 'max', 'min', 'last']).fillna(0)
    transactions.columns = transactions.columns.to_flat_index()
    transactions.to_csv('./data/static_network.csv')
    transactions = transactions.reset_index()
    return transactions

def load_labels():
    # Ground Truth Labels
    logging.info('Loading Labels...')
    labels = pd.read_csv(label_path, 
                        delimiter=',', 
                        )
    labels_set = set(labels['nodeId'])
    return labels, labels_set

def vertex_id_map(node_features, static_edge_features):
    # Vertex id map
    logging.info('Mapping vertex id to consecutive integers')
    nodes_set = set(np.unique([static_edge_features['srcId'], static_edge_features['dstId']]))
    node_features_set = set(node_features.index)
    feat_graph_intersec_set = node_features_set.intersection(nodes_set)
    logging.info('Number of node features in features.txt: {}'.format(len(node_features_set)))
    logging.info('Number of nodes in networks.csv: {}'.format(len(nodes_set)))
    logging.info('Number of node in the intersection: {}'.format(len(feat_graph_intersec_set)))
    
    number_of_nodes = len(feat_graph_intersec_set)
    v_mapping = dict(zip(list(feat_graph_intersec_set), range(number_of_nodes)))  # key is vertex id, value is new vertex id

    logging.info('Node id mapping created')
    logging.info('Save the mapping to {}'.format(vertex_map_path))
    with open(vertex_map_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in v_mapping.items():
            writer.writerow([key, value])
    logging.info(style.GREEN('Vertex id mapping is sucessfully saved to {}'.format(vertex_map_path)))

    return feat_graph_intersec_set, number_of_nodes, v_mapping


def preprocess_node_features(features, feat_graph_intersec_set, v_mapping):
    logging.info('Preprocessing node features')
    logging.info('Filtering nodes')
    features = features.loc[list(feat_graph_intersec_set)]
    logging.info('Mean imputation on the missing value')
    features = features.fillna(features.mean())
    # features = features.fillna(0)
    features.index = features.index.map(lambda x: v_mapping[x])
    features.sort_index(inplace=True)
    features = features.values

    # standardize node features and convert it to sparse matrix
    scaler = preprocessing.StandardScaler().fit(features)
    large_variance_column_index = np.where(scaler.var_ > 100)
    features[:, large_variance_column_index] = np.cbrt(features[:, large_variance_column_index])
    scaler = preprocessing.StandardScaler().fit(features)
    features = scaler.transform(features)
    logging.info('features shape: {}'.format(features.shape))
    # np.savetxt(processed_node_features_path, features, delimiter=",")
    # logging.info(style.GREEN('Processed node features is sucessfully saved to {}'.format(processed_node_features_path)))
    return features


def preprocess_labels(labels, feat_graph_intersec_set, v_mapping):
    logging.info('filtering unused nodes in the label')
    feat_graph_label_intersec_set = feat_graph_intersec_set.intersection(labels_set)
    labels = labels.set_index('nodeId')
    labels = labels.loc[list(feat_graph_label_intersec_set)]
    logging.info('mapping labels node id to new node id')
    labels.index = labels.index.map(lambda x: v_mapping[x])
    labels = labels.dropna(axis='rows')
    labels.index = labels.index.astype(int)
    labels.to_csv(processed_label_path)
    # convert label to one-hot format
    logging.info('convert label to one-hot format')
    one_hot_labels = pd.get_dummies(data=labels, dummy_na=True, columns=['label']) # N X (#edge attr)  # one hot 
    one_hot_labels = one_hot_labels.drop(['label_nan'], axis=1)
    logging.info('Train, validation, test split')
    # train, val, test split
    if os.path.exists(train_val_test_mask):
        logging.info('The mask file: {} exists! Reading train, val, test mask from the file'.format(train_val_test_mask))
        train_val_test_label = pd.read_csv(train_val_test_mask, delimiter=',', header=None, names=['id', 'mode'])
        train_val_test_label = train_val_test_label.set_index('id')
        train_val_test_label = train_val_test_label.loc[list(feat_graph_label_intersec_set)]
        train_val_test_label.index = train_val_test_label.index.map(lambda x: v_mapping[x])
        train_val_test_label = train_val_test_label.dropna(axis='rows')
        train_val_test_label.index = train_val_test_label.index.astype(int)
        train_id = np.array(list(set(train_val_test_label[train_val_test_label['mode'] == 'train'].index.values)))
        val_id = np.array(list(set(train_val_test_label[train_val_test_label['mode'] == 'val'].index.values)))
        test_id = np.array(list(set(train_val_test_label[train_val_test_label['mode'] == 'test'].index.values)))
    else:
        logging.info('The mask file: {} doest not exist. Performing train, val, test split'.format(train_val_test_mask))
        train_id, test_id, y_train, y_test = train_test_split(labels.index, labels['label'], 
                                            test_size=0.2, random_state=6, stratify=labels['label'])
        train_id, val_id, y_train, y_val = train_test_split(train_id, y_train, 
                                            test_size=0.2, random_state=6, stratify=y_train)
    train_mask = np.zeros((number_of_nodes,)).astype(int)
    val_mask = np.zeros((number_of_nodes,)).astype(int)
    test_mask = np.zeros((number_of_nodes,)).astype(int)

    # train_ratio = 0.8
    np.random.seed(1)
    train_mask[list(train_id)] = 1
    val_mask[list(val_id)] = 1
    test_mask[list(test_id)] = 1

    # one_hot_labels = one_hot_labels.values[:,:-1]  # convert to numpy format and remove the nan column
    y = np.zeros(number_of_nodes)
    y[one_hot_labels.index] = np.argmax(one_hot_labels.values, 1)

    logging.info('train_mask shape: {}'.format(train_mask.shape))
    logging.info('val_mask shape: {}'.format(val_mask.shape))
    logging.info('test_mask shape: {}'.format(test_mask.shape))
    return train_id, val_id, test_id

def load_graph(node_features, edge_features, feat_graph_intersec_set, number_of_nodes, v_mapping):
    logging.info('Exporting dgl graph')
    edge_features = edge_features[np.in1d(edge_features['srcId'], list(feat_graph_intersec_set))]
    edge_features = edge_features[np.in1d(edge_features['dstId'], list(feat_graph_intersec_set))]
    logging.info('*** Number of edges after filtering : {}'.format(edge_features.shape[0]))
    edge_from_id = edge_features['srcId'].values.astype(int)
    edge_to_id = edge_features['dstId'].values.astype(int)
    # Map vertex id to consecutive integers
    edge_from_id = np.vectorize(v_mapping.get)(edge_from_id)
    edge_to_id = np.vectorize(v_mapping.get)(edge_to_id)
    edge_features = edge_features.iloc[:,2:].values
    # Create DGL Graph
    g = dgl.DGLGraph()
    g.add_nodes(number_of_nodes)
    g.add_edges(u=edge_from_id, 
                v=edge_to_id, 
                data={'edge_features': torch.from_numpy(edge_features)})
    logging.info(g.edge_attr_schemes())

    means = g.edata['edge_features'].mean(dim=1, keepdim=True)
    stds = g.edata['edge_features'].std(dim=1, keepdim=True)
    g.edata['edge_features'] = (g.edata['edge_features'] - means) / stds
    g.edata['edge_features'] = g.edata['edge_features'].to(dtype=torch.float32)
    logging.info('Adding loop')
    num_edge_feats = edge_features.shape[1]
    g.add_edges(g.nodes(), g.nodes(), 
            data={'edge_features': torch.zeros((g.number_of_nodes(), num_edge_feats))})
    g.ndata['node_features'] = torch.FloatTensor(node_features)

    with open(dgl_pickle_path, 'wb') as f:
        pickle.dump(g, f)
    logging.info(style.GREEN('DGL Graph is sucessfully saved to {}'.format(dgl_pickle_path)))
    return g


node_features = load_node_features()
static_edge_features = hand_craft_edge_features()
labels, labels_set = load_labels()
feat_graph_intersec_set, number_of_nodes, v_mapping = vertex_id_map(node_features, static_edge_features)
node_features = preprocess_node_features(node_features, feat_graph_intersec_set, v_mapping)
train_id, val_id, test_id = preprocess_labels(labels, feat_graph_intersec_set, v_mapping)
g = load_graph(node_features, static_edge_features, feat_graph_intersec_set, number_of_nodes, v_mapping)
logging.info(style.GREEN('Success'))

