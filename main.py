import torch
import argparse
import torch.nn.functional as F
import sys
import os
import logging
from dgl import DGLGraph
from dmt.models import EdgePropGAT, GAT, GAT_EdgeAT, MiniBatchEdgeProp, MiniBatchEdgePropInfer, MiniBatchGCNInfer, MiniBatchGCNSampling, MiniBatchGraphSAGEInfer, MiniBatchGraphSAGESampling
from dmt.models import MiniBatchEdgePropPlus, MiniBatchEdgePropPlusInfer
from dmt.models.unsupervised import DGI, MiniBatchDGI
from dmt.trainer import Trainer
from dmt.mini_batch_trainer import MiniBatchTrainer
from dmt.unsupervised_trainer import UnsupervisedTrainer
from dmt.unsupervised_mini_batch_trainer import UnsupervisedMiniBatchTrainer
from dmt.classical_baseline_trainer import ClassicalBaseline
# from dmt.data import register_data_args, load_data
from data.data_loader import Dataset
from dmt.utils import Params, set_logger, style
import numpy as np

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

def main(params):
    # load and preprocess dataset
    data = Dataset(data_path=params.data_dir, 
                   directed=False, 
                   k=params.k)
    features = data.g.ndata['node_features']
    logging.info('features shape: {}'.format(features.shape))
    labels = data.labels
    train_id = data.train_id
    val_id = data.val_id
    test_id = data.test_id
    # train_mask = torch.ByteTensor(data.train_mask)
    # val_mask = torch.ByteTensor(data.val_mask)
    # test_mask = torch.ByteTensor(data.test_mask)
    num_feats = features.shape[1]
    # num_edge_feats = edge_features.shape[1]
    num_edge_feats = data.num_edge_feats
    n_classes = data.num_classes
    n_edges = data.g.number_of_edges()
    n_nodes = data.g.number_of_nodes()
    logging.info("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           len(train_id),
           len(val_id),
           len(test_id)))
    if params.gpu < 0:
        cuda = False
        cuda_context = None
    else:
        cuda = True
        torch.cuda.set_device(params.gpu)
        cuda_context = torch.device('cuda:{}'.format(params.gpu))

    # create DGL graph
    g = data.g
    n_edges = g.number_of_edges()
    # add self loop
    # print(g.edata)
    #g.add_edges(g.nodes(), g.nodes(), data={'edge_features': torch.zeros((n_nodes, num_edge_feats))})

    # create model
    if params.model == "EdgePropAT":
        heads = [params.num_heads] * params.num_layers
        model = EdgePropGAT(g,
                    params.num_layers,
                    num_feats,
                    num_edge_feats, 
                    params.num_hidden,
                    n_classes,
                    heads,
                    F.elu,
                    params.in_drop,
                    params.attn_drop,
                    params.alpha,
                    params.residual, 
                    params.use_batch_norm)
        
    elif params.model == "GAT":
        heads = [params.num_heads] * params.num_layers
        model = GAT(g,
                    params.num_layers,
                    num_feats,
                    params.num_hidden,
                    n_classes,
                    heads,
                    F.elu,
                    params.in_drop,
                    params.attn_drop,
                    params.alpha,
                    params.residual)
        
    elif params.model == "GAT_EdgeAT":
        heads = [params.num_heads] * params.num_layers
        model = GAT_EdgeAT(g,
                    params.num_layers,
                    num_feats,
                    num_edge_feats, 
                    params.num_hidden,
                    n_classes,
                    heads,
                    F.elu,
                    params.in_drop,
                    params.attn_drop,
                    params.alpha,
                    params.residual)
        
    elif params.model == "MiniBatchEdgeProp":
        model = MiniBatchEdgeProp(
                    g, 
                    params.num_layers,
                    num_feats,
                    num_edge_feats, 
                    params.num_hidden,
                    n_classes,
                    F.elu,
                    params.in_drop,
                    cuda)
        model_infer = MiniBatchEdgePropInfer(
                    g, 
                    params.num_layers,
                    num_feats,
                    num_edge_feats, 
                    params.num_hidden,
                    n_classes,
                    F.elu,
                    0, #params.in_drop,
                    cuda)
        
    elif params.model == 'MiniBatchGCN':
        model = MiniBatchGCNSampling(
                    in_feats=num_feats,
                    n_hidden=params.num_hidden,
                    n_classes=n_classes,
                    n_layers=params.num_layers,
                    activation=F.relu,
                    dropout=params.in_drop
        )
        model_infer = MiniBatchGCNInfer(
                    in_feats=num_feats,
                    n_hidden=params.num_hidden,
                    n_classes=n_classes,
                    n_layers=params.num_layers,
                    activation=F.relu
        )
        
    elif params.model == 'MiniBatchGraphSAGE':
        model = MiniBatchGraphSAGESampling(
                    in_feats=num_feats,
                    n_hidden=params.num_hidden,
                    n_classes=n_classes,
                    n_layers=params.num_layers,
                    activation=F.relu,
                    dropout=params.in_drop
        )
        model_infer = MiniBatchGraphSAGEInfer(
                    in_feats=num_feats,
                    n_hidden=params.num_hidden,
                    n_classes=n_classes,
                    n_layers=params.num_layers,
                    activation=F.relu
        )
        
    elif params.model == 'MiniBatchEdgePropPlus':
        model = MiniBatchEdgePropPlus(
                    g, 
                    params.num_layers,
                    num_feats,
                    num_edge_feats, 
                    params.node_hidden_dim,
                    params.edge_hidden_dim,
                    params.fc_hidden_dim,
                    n_classes,
                    F.elu,
                    params.in_drop,
                    params.residual, 
                    params.use_batch_norm)        

        model_infer = MiniBatchEdgePropPlusInfer(
                    g, 
                    params.num_layers,
                    num_feats,
                    num_edge_feats, 
                    params.node_hidden_dim,
                    params.edge_hidden_dim,
                    params.fc_hidden_dim,
                    n_classes,
                    F.elu,
                    0, #params.in_drop,
                    params.residual, 
                    params.use_batch_norm)

    elif params.model == 'DGI':
        print(g.ndata)
        unsupervised_model = DGI.DGI(
                    g=g, 
                    conv_model=params.conv_model, 
                    in_feats=num_feats, 
                    n_hidden=params.node_hidden_dim, 
                    n_layers=params.num_layers, 
                    activation=F.relu, 
                    dropout=params.in_drop)
        encoder = unsupervised_model.encoder
    elif params.model == 'MiniBatchDGI':
        print(g.ndata)
        unsupervised_model = MiniBatchDGI.DGI(
                    g=g, 
                    conv_model=params.conv_model, 
                    in_feats=num_feats, 
                    edge_in_feats=num_edge_feats,
                    n_hidden=params.node_hidden_dim, 
                    n_layers=params.num_layers, 
                    activation=F.relu, 
                    dropout=params.in_drop,
                    cuda=cuda)
        unsupervised_model_infer = MiniBatchDGI.DGIInfer(
                    g=g, 
                    conv_model=params.conv_model, 
                    in_feats=num_feats, 
                    edge_in_feats=num_edge_feats,
                    n_hidden=params.node_hidden_dim, 
                    n_layers=params.num_layers, 
                    activation=F.relu, 
                    dropout=params.in_drop, 
                    cuda=cuda)
        encoder = unsupervised_model.encoder
        encoder_infer = unsupervised_model_infer.encoder
        # decoder = DGI.Classifier(
        #             params.node_hidden_dim, 
        #             n_classes)
    elif params.model.lower() == 'classical_baselines':
        pass
    else:
        logging.info('The model \"{}\" is not implemented'.format(params.model))
        sys.exit(0)

    # weights = len(train_id) / (n_classes * np.bincount(labels.loc[train_id]['label'].values))
    # class_weights = torch.FloatTensor(weights).cuda()
    # loss_fcn = torch.nn.CrossEntropyLoss(weight=class_weights)
    if not params.model.lower() == 'classical_baselines':
        loss_fcn = torch.nn.CrossEntropyLoss()
        if params.model.lower() in ['dgi', 'minibatchdgi']:
            if cuda:
                unsupervised_model.cuda()   
                if 'model_infer' in locals():
                    model_infer.cuda()
                elif 'unsupervised_model_infer' in locals():
                    unsupervised_model_infer.cuda()
            logging.info(unsupervised_model)
            unsupervised_optimizer = torch.optim.Adam(unsupervised_model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
            # logging.info(decoder)
        else:
            if cuda:
                model.cuda()   
                if 'model_infer' in locals():
                    model_infer.cuda()
            logging.info(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    if params.model.lower() in ['minibatchedgeprop', 'minibatchgcn', 'minibatchgraphsage', 'minibatchedgepropplus']:
        g.readonly()
        # initialize the history for control variate
        # see control variate in https://arxiv.org/abs/1710.10568
        # for i in range(params.num_layers):
        #     #g.ndata['history_{}'.format(i)] = torch.zeros((features.shape[0], params.node_hidden_dim))
        #     g.ndata['history_{}'.format(i)] = torch.zeros((features.shape[0], params.num_hidden))
        #g.edata['edge_features'] = data.graph.edata['edge_features']
        norm = 1./g.in_degrees().unsqueeze(1).float()
        g.ndata['norm'] = norm
        print('graph node features', g.ndata['node_features'].shape)
        print('graph edge features', g.edata['edge_features'].shape)

        degs = g.in_degrees().numpy()
        degs[degs > params.num_neighbors] = params.num_neighbors
        g.ndata['subg_norm'] = torch.FloatTensor(1./degs).unsqueeze(1)  # for calculating P_hat

        trainer = MiniBatchTrainer(
                        g=g, 
                        model=model, 
                        model_infer=model_infer,
                        loss_fn=loss_fcn, 
                        optimizer=optimizer, 
                        epochs=params.epochs, 
                        features=features, 
                        labels=labels,
                        train_id=train_id,
                        val_id=val_id,
                        test_id=test_id, 
                        fast_mode=params.fastmode, 
                        n_edges=n_edges, 
                        patience=params.patience, 
                        batch_size=params.batch_size, 
                        test_batch_size=params.test_batch_size, 
                        num_neighbors=params.num_neighbors, 
                        n_layers=params.num_layers, 
                        model_dir=params.model_dir, 
                        num_cpu=params.num_cpu, 
                        cuda_context=cuda_context)
    elif params.model.lower() in ['minibatchdgi']:
        g.readonly()
        # initialize the history for control variate
        #g.edata['edge_features'] = data.graph.edata['edge_features']
        norm = 1./g.in_degrees().unsqueeze(1).float()
        g.ndata['norm'] = norm
        print('graph node features', g.ndata['node_features'].shape)
        print('graph edge features', g.edata['edge_features'].shape)

        degs = g.in_degrees().numpy()
        degs[degs > params.num_neighbors] = params.num_neighbors
        g.ndata['subg_norm'] = torch.FloatTensor(1./degs).unsqueeze(1)  # for calculating P_hat

        trainer = UnsupervisedMiniBatchTrainer(
                        g=g, 
                        unsupervised_model=unsupervised_model, 
                        unsupervised_model_infer=unsupervised_model_infer,
                        encoder=encoder,
                        encoder_infer=encoder_infer, 
                        loss_fn=loss_fcn, 
                        optimizer=unsupervised_optimizer, 
                        epochs=params.epochs, 
                        features=features, 
                        labels=labels, 
                        train_id=train_id, 
                        val_id=val_id, 
                        test_id=test_id, 
                        fast_mode=params.fastmode, 
                        n_edges=n_edges, 
                        patience=params.patience, 
                        batch_size=params.batch_size, 
                        test_batch_size=params.test_batch_size, 
                        num_neighbors=params.num_neighbors, 
                        n_layers=params.num_layers, 
                        model_dir=params.model_dir, 
                        num_cpu=params.num_cpu, 
                        cuda_context=cuda_context)
    elif params.model.lower() in ['dgi']:
        trainer = UnsupervisedTrainer(
                        g=g, 
                        unsupervised_model=unsupervised_model,
                        encoder=encoder,
                        decoder=None,
                        loss_fn=loss_fcn, 
                        unsupervised_optimizer=unsupervised_optimizer,
                        decoder_optimizer=None, 
                        epochs=params.epochs,
                        features=features,
                        labels=labels,
                        train_id=train_id,
                        val_id=val_id, 
                        test_id=test_id, 
                        fast_mode=params.fastmode, 
                        n_edges=n_edges, 
                        patience=params.patience, 
                        n_layers=params.num_layers, 
                        model_dir=params.model_dir, 
                        num_cpu=params.num_cpu, 
                        cuda_context=cuda_context)
    elif params.model.lower() in ['classical_baselines']:
        trainer = ClassicalBaseline(
                        features=features,
                        labels=labels,
                        train_id=train_id,
                        val_id=val_id,
                        test_id=test_id
            )
    else:
        logging.info(style.RED('The model: {} is not supported yet.'.format(params.model)))
        sys.exit(0)
        # if cuda:
        #     g.edata['edge_features'] = data.g.edata['edge_features'].cuda()
        # trainer = Trainer(
        #                 model=model, 
        #                 loss_fn=loss_fcn, 
        #                 optimizer=optimizer, 
        #                 epochs=params.epochs, 
        #                 features=features, 
        #                 labels=labels, 
        #                 train_mask=train_mask, 
        #                 val_mask=val_mask, 
        #                 test_mask=test_mask, 
        #                 fast_mode=params.fast_mode, 
        #                 n_edges=n_edges, 
        #                 patience=params.patience, 
        #                 model_dir=params.model_dir)
    trainer.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Examples')
    # register_data_args(parser)
    parser.add_argument("--data-dir", type=str, required=True, 
                        help="Directory containing network.csv, features.csv, labels.csv")
    parser.add_argument("--model-dir", type=str, required=True, 
                        help="Directory containing params.json")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'
    parser.add_argument('--gpu', type=int, default=None, required=True, 
                        help="gpu id, -1 if cpu")
    parser.add_argument('--k', type=int, default=None, required=True, help="k")
    args = parser.parse_args()


    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info(args)

    # load params
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    params.data_dir = args.data_dir
    params.model_dir = args.model_dir
    params.gpu = args.gpu
    params.k=args.k

    # params.cuda = torch.cuda.is_available()

    main(params)
