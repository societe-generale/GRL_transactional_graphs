#!Copyright (c) 2022, Société Générale.
#!All rights reserved.

#!This source code is licensed under the BSD 2-clauses license found in the
#!LICENSE file in the root directory of this source tree.  

import torch as th
import torch.nn as nn

import dgl.function as fn
from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv

from gensim.models.word2vec import Word2Vec
import multiprocessing as mp
import numpy as np
from itertools import repeat

from utils.deepwalk_utils import generate_deepwalk_rw

ACTIVATIONS = {
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "identity": nn.Identity()
}


RW_GENERATOR = {
    "deepwalk": generate_deepwalk_rw
}


class DeepWalk:
    """An implementation of DeepWalk.
    Args:
        walk_number (int): number of random walks. Defaults to 10.
        walk_length (int): length of random walks. Defaults to  80.
        dimensions (int): dimensionality of embedding. Defaults to  128.
        window_size (int): matrix power order. Defaults to  5.
        epochs (int): number of epochs. Defaults to  1.
    """
    def __init__(self,
                 dimensions: int = 128,
                 walk_number: int = 10,
                 walk_length: int = 80,
                 window_size: int = 5,
                 epochs: int = 1,
                 *args,
                 **kwargs):
        self.dimensions = dimensions
        self.n_generated_walks = None
        self.walk_number = walk_number
        self.walk_length = walk_length
        self.window_size = window_size
        self.epochs = epochs

    def _skip_gram(self, walks):
        model = Word2Vec(walks,
                         size=self.dimensions,
                         window=self.window_size,
                         iter=self.epochs,
                         min_count=0,
                         sg=1,
                         hs=1,
                         workers=int(round(mp.cpu_count() * 0.3)),
                         seed=0)
        return model.wv

    def _generate_walks(self, graph):
        nodes = list(graph) * self.walk_number
        with mp.Pool(processes=int(round(mp.cpu_count() * 0.3))) as pool:
            res = pool.starmap_async(func=generate_deepwalk_rw,
                                     iterable=zip(repeat(graph), nodes,
                                                  repeat(self.walk_length)))
            walks = res.get()
        return walks

    def fit(self, graph, **kwargs):
        # Generate random walks
        walks = self._generate_walks(graph)
        np.random.shuffle(walks)

        # Compute the embedding by training Word2Vec
        walks = [[str(w) for w in walk] for walk in walks]
        wv = self._skip_gram(walks)

        self.n_generated_walks = len(walks)
        embeddings = wv.vectors
        node2id = {word: index for index, word in enumerate(wv.index2word)}
        self.ordered_embeddings = np.array(
            [embeddings[node2id[str(node)]] for node in graph])
        return self

    def get_embedding(self):
        return np.array(self.ordered_embeddings)


class GAT(nn.Module):
    """
    GAT implementation derived from:
    https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/gat.py
    """
    def __init__(self, g, num_layers, in_dim, num_hidden, num_classes, heads,
                 activation, feat_drop, attn_drop, negative_slope, residual,
                 allow_zero_in_degree):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = ACTIVATIONS[activation]

        self.gat_layers.append(
            GATConv(in_dim,
                    num_hidden,
                    heads[0],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    False,
                    self.activation,
                    allow_zero_in_degree=allow_zero_in_degree))

        for lay in range(1, num_layers - 1):
            self.gat_layers.append(
                GATConv(num_hidden * heads[lay - 1],
                        num_hidden,
                        heads[lay],
                        feat_drop,
                        attn_drop,
                        negative_slope,
                        residual,
                        self.activation,
                        allow_zero_in_degree=allow_zero_in_degree))

        self.gat_layers.append(
            GATConv(num_hidden * heads[-2],
                    num_classes,
                    heads[-1],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    None,
                    allow_zero_in_degree=allow_zero_in_degree))

    def forward(self, inputs):
        h = inputs
        for lay in range(self.num_layers - 1):
            h = self.gat_layers[lay](self.g, h).flatten(1)
        self.embeddings = h
        logits = self.gat_layers[-1](self.g, h).mean(1)

        return logits


class GAT_stochastic(GAT):
    def __init__(self, g, num_layers, in_dim, num_hidden, num_classes, heads,
                 activation, feat_drop, attn_drop, negative_slope, residual,
                 allow_zero_in_degree):
        super(GAT_stochastic,
              self).__init__(g, num_layers, in_dim, num_hidden, num_classes,
                             heads, activation, feat_drop, attn_drop,
                             negative_slope, residual, allow_zero_in_degree)

    def forward(self, mfgs, inputs):
        h_dst = inputs[:mfgs[0].num_dst_nodes()]
        h = self.gat_layers[0](mfgs[0], (inputs, h_dst)).flatten(1)
        for i, layer in enumerate(self.gat_layers[1:-1]):
            h_dst = h[:mfgs[i + 1].num_dst_nodes()]
            h = layer(mfgs[i + 1], (h, h_dst)).flatten(1)
        h_dst = h[:mfgs[-1].num_dst_nodes()]
        logits = self.gat_layers[-1](mfgs[-1], (h, h_dst)).mean(1)

        return logits


class GCN(nn.Module):
    """
    GNC implementation derived from:
    https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/gcn.py
    """
    def __init__(self, g, num_layers, in_dim, num_hidden, num_classes,
                 activation, dropout, allow_zero_in_degree):
        super(GCN, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = ACTIVATIONS[activation]

        self.layers.append(
            GraphConv(in_dim,
                      num_hidden,
                      activation=self.activation,
                      allow_zero_in_degree=allow_zero_in_degree))

        for _ in range(1, num_layers - 1):
            self.layers.append(
                GraphConv(num_hidden,
                          num_hidden,
                          activation=self.activation,
                          allow_zero_in_degree=allow_zero_in_degree))

        self.layers.append(
            GraphConv(num_hidden,
                      num_classes,
                      allow_zero_in_degree=allow_zero_in_degree))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            if i == len(self.layers):
                self.embeddings = h
            h = layer(self.g, h)

        return h


class GCN_stochastic(GCN):
    def __init__(self, g, num_layers, in_dim, num_hidden, num_classes,
                 activation, dropout, allow_zero_in_degree):
        super(GCN_stochastic,
              self).__init__(g, num_layers, in_dim, num_hidden, num_classes,
                             activation, dropout, allow_zero_in_degree)

    def forward(self, mfgs, features):
        h_dst = features[:mfgs[0].num_dst_nodes()]
        h = self.layers[0](mfgs[0], (features, h_dst))
        for i, layer in enumerate(self.layers[1:]):
            h_dst = h[:mfgs[i + 1].num_dst_nodes()]
            h_dst = self.dropout(h_dst)
            h = layer(mfgs[i + 1], (h, h_dst))

        return h


class GraphSAGE(nn.Module):
    """
    DGL implementation of GraphSAGE
    """
    def __init__(self, g, num_layers, in_dim, num_hidden, num_classes,
                 activation, dropout, aggregator_type):
        super(GraphSAGE, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = ACTIVATIONS[activation]

        self.layers.append(
            SAGEConv(in_dim,
                     num_hidden,
                     aggregator_type,
                     activation=self.activation))

        for _ in range(1, num_layers - 1):
            self.layers.append(
                SAGEConv(num_hidden,
                         num_hidden,
                         aggregator_type,
                         activation=self.activation))

        self.layers.append(SAGEConv(num_hidden, num_classes, aggregator_type))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            if i == len(self.layers):
                self.embeddings = h
            h = layer(self.g, h)

        return h


class GraphSAGE_stochastic(GraphSAGE):
    def __init__(self, g, num_layers, in_dim, num_hidden, num_classes,
                 activation, dropout, aggregator_type):
        super(GraphSAGE_stochastic,
              self).__init__(g, num_layers, in_dim, num_hidden, num_classes,
                             activation, dropout, aggregator_type)

    def forward(self, mfgs, features):
        h_dst = features[:mfgs[0].num_dst_nodes()]
        h = self.layers[0](mfgs[0], (features, h_dst))
        for i, layer in enumerate(self.layers[1:]):
            h_dst = h[:mfgs[i + 1].num_dst_nodes()]
            h_dst = self.dropout(h_dst)
            h = layer(mfgs[i + 1], (h, h_dst))

        return h


class GCN_LP(nn.Module):
    """
    GCN Implementation for link prediction task.
    Contains GCN encoder and add a dot product predictor.
    """
    def __init__(self, g, num_layers, in_dim, num_hidden, dim_embedding,
                 activation, dropout, allow_zero_in_degree):
        super(GCN_LP, self).__init__()
        self.encoder = GCN(g, num_layers, in_dim, num_hidden, dim_embedding,
                           activation, dropout, allow_zero_in_degree)
        self.pred = MLPPredictor(2 * dim_embedding, 1)

    def forward(self, g, x):
        h = self.encoder(x)
        self.embeddings = h
        edge = g.edges()
        emb = th.cat([h[edge[0]], h[edge[1]]], dim=1)
        edge_score = self.pred(emb)
        return th.sigmoid(edge_score)


class GAT_LP(nn.Module):
    """
    GAT Implementation for link prediction task.
    Contains GAT encoder and add a dot product predictor.
    """
    def __init__(self, g, num_layers, in_dim, num_hidden, dim_embedding, heads,
                 activation, feat_drop, attn_drop, negative_slope, residual,
                 allow_zero_in_degree):
        super(GAT_LP, self).__init__()
        self.encoder = GAT(g, num_layers, in_dim, num_hidden, dim_embedding,
                           heads, activation, feat_drop, attn_drop,
                           negative_slope, residual, allow_zero_in_degree)
        self.pred = MLPPredictor(2 * dim_embedding, 1)

    def forward(self, g, x):
        h = self.encoder(x)
        self.embeddings = h
        edge = g.edges()
        emb = th.cat([h[edge[0]], h[edge[1]]], dim=1)
        edge_score = self.pred(emb)
        return th.sigmoid(edge_score)


class GraphSAGE_LP(nn.Module):
    def __init__(self, g, num_layers, in_dim, num_hidden, dim_embedding,
                 activation, dropout, aggregator_type):
        super(GraphSAGE_LP, self).__init__()
        self.encoder = GraphSAGE(g, num_layers, in_dim, num_hidden,
                                 dim_embedding, activation, dropout,
                                 aggregator_type)
        self.pred = MLPPredictor(2 * dim_embedding, 1)

    def forward(self, g, x):
        h = self.encoder(x)
        self.embeddings = h
        edge = g.edges()
        emb = th.cat([h[edge[0]], h[edge[1]]], dim=1)
        edge_score = self.pred(emb)
        return th.sigmoid(edge_score)


class MLPPredictor(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=3, activation="relu"):
        super(MLPPredictor, self).__init__()
        self.num_layers = num_layers
        self.activation = ACTIVATIONS[activation]
        fc_layers = [
            nn.Linear(in_dim // 2**lay, in_dim // 2**(lay + 1), bias=True)
            for lay in range(num_layers - 1)
        ]
        fc_layers.append(
            nn.Linear(in_dim // 2**(num_layers - 1), out_dim, bias=True))
        self.fc_layers = nn.ModuleList(fc_layers)

    def forward(self, features):
        h = features
        for _, layer in enumerate(self.fc_layers[:-1]):
            h = layer(h)
            h = self.activation(h)
        outputs = self.fc_layers[self.num_layers - 1](h)
        return outputs


class InnerProductPredictor(nn.Module):
    def __init__(self, in_dim):
        super(InnerProductPredictor, self).__init__()
        self.in_dim = in_dim

    def forward(self, features):
        emb_size = int(self.in_dim / 2)
        h_i, h_j = features[:, :emb_size], features[:, emb_size:]
        dot_product = th.bmm(h_i.view((h_i.size()[0], 1, emb_size)),
                             h_j.view((h_j.size()[0], emb_size, 1)))
        return dot_product.view((dot_product.size()[0], 1))
