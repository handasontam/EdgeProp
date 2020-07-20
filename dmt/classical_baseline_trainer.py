from .utils.metrics import torch_accuracy, accuracy, micro_f1, macro_f1, hamming_loss, micro_precision, micro_recall, macro_precision, macro_recall
from .utils.torch_utils import EarlyStopping
import torch
try:
    from tensorboardX import SummaryWriter
    use_tensorboardx = True
except:
    use_tensorboardx = False
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import os
from dgl.contrib.sampling import NeighborSampler
import dgl.function as fn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


class ClassicalBaseline(object):
    def __init__(self, features, labels, train_id, val_id, test_id):
        self.train_id = train_id
        self.val_id = val_id
        self.test_id = test_id
        self.features = features
        self.labels = labels

        # initialize early stopping object

    def train(self):
        classifiers = {
                       'Logistic': LogisticRegression(solver='lbfgs', multi_class='ovr', class_weight='balanced'), 
                       'LinearSVC': LinearSVC(class_weight='balanced'), 
                       'DecisionTree': DecisionTreeClassifier(class_weight='balanced'), 
                       'GradientBoosting': GradientBoostingClassifier(), 
                       'RandomForest': RandomForestClassifier(class_weight='balanced'), 
                        }
        features = self.features.detach().cpu().numpy()

        X_train = features[self.train_id]
        y_train = self.labels.loc[self.train_id]['label']

        X_test = features[self.test_id]
        y_test = self.labels.loc[self.test_id]['label']
        
        for name, classifier in classifiers.items():
            classifier.fit(X=X_train, y=y_train)
            pred = classifier.predict(X_test)
            val_macro_precision = macro_precision(pred, y_test)
            val_macro_recall = macro_recall(pred, y_test)
            val_micro_f1 = micro_f1(pred, y_test)
            val_macro_f1 = macro_f1(pred, y_test)
            logging.info('{}(precision) : {}'.format(name, val_macro_precision))
            logging.info('{}(recall) : {}'.format(name, val_macro_recall))
            logging.info('{}(macro f1) : {}'.format(name, val_macro_f1))
            logging.info('{}(micro f1) : {}'.format(name, val_micro_f1))
            if name == 'GradientBoosting':
                importances = classifier.feature_importances_
                indices = np.argsort(importances)[::-1]

                # Print the feature ranking
                logging.info("Feature ranking:")

                for f in range(features.shape[1]):
                    logging.info("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

                # feature_importance = classifier.feature_importances_
                # # make importances relative to max importance
                # feature_importance = 100.0 * (feature_importance / feature_importance.max())
                # sorted_idx = np.argsort(feature_importance)
                # pos = np.arange(sorted_idx.shape[0]) + .5
                # plt.subplot(1, 2, 2)
                # plt.barh(pos, feature_importance[sorted_idx], align='center')
                # plt.yticks(pos, sorted_idx)
                # plt.xlabel('Relative Importance')
                # plt.title('Variable Importance')
                # plt.savefig(os.path.join(self.model_dir, 'features_importance.png'))
