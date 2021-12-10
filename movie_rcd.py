#!/usr/bin/env python
# -*- coding:utf-8 -*-
from flask import Flask
from flask import render_template#The template engine, renders templates
from flask import request
from flask_restful import Api,Resource
import pandas as pd
import numpy as np
import json
from scipy.linalg import sqrtm 
from scipy.special import softmax
import networkx as nx
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
import matplotlib.pyplot as plt
from matplotlib import animation
#%matplotlib inline
from IPython.display import HTML
import math
from operator import itemgetter

app = Flask(__name__)
api = Api(app)

@app.route('/')
def index():
    return render_template('index.html')

# save the byte->json problem
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)

@app.route('/Movie_RCD', methods = ['GET', 'POST'])
def useremo():
    # draw picture
    def draw_kkl(nx_G, label_map, node_color, pos=None, **kwargs):
        fig, ax = plt.subplots(figsize=(10,10))
        if pos is None:
            pos = nx.spring_layout(nx_G, k=5/np.sqrt(nx_G.number_of_nodes()))
        nx.draw(
            nx_G, pos, with_labels=label_map is not None, 
            labels=label_map, 
            node_color=node_color, 
            ax=ax, **kwargs)

    # generate graph
    def generate_g(num):
        df_base = pd.read_csv('ml-latest-small/ratings.csv')
        df_rate = df_base[['userId','movieId','rating']].reset_index(drop = True)
        df_love = df_rate[df_rate['rating'] == 5].reset_index(drop = True)
        df_love = df_love[df_love['userId']<num].reset_index(drop = True)
        df_relation = df_love.merge(df_love, on='movieId', how='inner')[['userId_x', 'userId_y', 'movieId']]
        df_relation = df_relation[df_relation['userId_x'] != df_relation['userId_y']].reset_index(drop = True)
        all_members = set(range(num-1))
        G = nx.Graph()
        G.add_nodes_from(all_members)
        G.name = 'ml_latest_small'
        for i in range(int(df_relation.shape[0])):
            if df_relation.iloc[i]['userId_x'] != df_relation.iloc[i]['userId_y']:
                row = df_relation.iloc[i]['userId_x']-1
                col = df_relation.iloc[i]['userId_y']-1
                G.add_edge(row, col)
                i = i + 1
        return G, df_love

    def glorot_init(nin, nout):
        sd = np.sqrt(6.0 / (nin + nout))
        return np.random.uniform(-sd, sd, size=(nin, nout))


    def xent(pred, labels):
        return -np.log(pred)[np.arange(pred.shape[0]), np.argmax(labels, axis=1)]


    def norm_diff(dW, dW_approx):
        return np.linalg.norm(dW - dW_approx) / (np.linalg.norm(dW) + np.linalg.norm(dW_approx))

    class GradDescentOptim():
        def __init__(self, lr, wd):
            self.lr = lr
            self.wd = wd
            self._y_pred = None
            self._y_true = None
            self._out = None
            self.bs = None
            self.train_nodes = None
            
        def __call__(self, y_pred, y_true, train_nodes=None):
            self.y_pred = y_pred
            self.y_true = y_true
            
            if train_nodes is None:
                self.train_nodes = np.arange(y_pred.shape[0])
            else:
                self.train_nodes = train_nodes
                
            self.bs = self.train_nodes.shape[0]
            
        @property
        def out(self):
            return self._out
        
        @out.setter
        def out(self, y):
            self._out = y
        

    class GCNLayer():
        def __init__(self, n_inputs, n_outputs, activation=None, name=''):
            self.n_inputs = n_inputs
            self.n_outputs = n_outputs
            self.W = glorot_init(self.n_outputs, self.n_inputs)
            self.activation = activation
            self.name = name
            
        def __repr__(self):
            return f"GCN: W{'_'+self.name if self.name else ''} ({self.n_inputs}, {self.n_outputs})"
            
        def forward(self, A, X, W=None):
            """
            Assumes A is (bs, bs) adjacency matrix and X is (bs, D), 
                where bs = "batch size" and D = input feature length
            """
            self._A = A
            self._X = (A @ X).T # for calculating gradients.  (D, bs)
            
            if W is None:
                W = self.W
            
            H = W @ self._X # (h, D)*(D, bs) -> (h, bs)
            if self.activation is not None:
                H = self.activation(H)
            self._H = H # (h, bs)
            return self._H.T # (bs, h)
    
        def backward(self, optim, update=True):
            dtanh = 1 - np.asarray(self._H.T)**2 # (bs, out_dim)
            d2 = np.multiply(optim.out, dtanh)  # (bs, out_dim) *element_wise* (bs, out_dim)
            
            self.grad = self._A @ d2 @ self.W # (bs, bs)*(bs, out_dim)*(out_dim, in_dim) = (bs, in_dim)     
            optim.out = self.grad
            
            dW = np.asarray(d2.T @ self._X.T) / optim.bs  # (out_dim, bs)*(bs, D) -> (out_dim, D)
            dW_wd = self.W * optim.wd / optim.bs # weight decay update
            
            if update:
                self.W -= (dW + dW_wd) * optim.lr 
            
            return dW + dW_wd

    
    class SoftmaxLayer():
        def __init__(self, n_inputs, n_outputs, name=''):
            self.n_inputs = n_inputs
            self.n_outputs = n_outputs
            self.W = glorot_init(self.n_outputs, self.n_inputs)
            self.b = np.zeros((self.n_outputs, 1))
            self.name = name
            self._X = None # Used to calculate gradients
            
        def __repr__(self):
            return f"Softmax: W{'_'+self.name if self.name else ''} ({self.n_inputs}, {self.n_outputs})"
        
        def shift(self, proj):
            shiftx = proj - np.max(proj, axis=0, keepdims=True)
            exps = np.exp(shiftx)
            return exps / np.sum(exps, axis=0, keepdims=True)
            
        def forward(self, X, W=None, b=None):
            """Compute the softmax of vector x in a numerically stable way.
            
            X is assumed to be (bs, h)
            """
            self._X = X.T
            if W is None:
                W = self.W
            if b is None:
                b = self.b

            proj = np.asarray(W @ self._X) + b # (out, h)*(h, bs) = (out, bs)
            return self.shift(proj).T # (bs, out)
        
        def backward(self, optim, update=True):
            # should take in optimizer, update its own parameters and update the optimizer's "out"
            # Build mask on loss
            train_mask = np.zeros(optim.y_pred.shape[0])
            train_mask[optim.train_nodes] = 1
            train_mask = train_mask.reshape((-1, 1))
            
            # derivative of loss w.r.t. activation (pre-softmax)
            d1 = np.asarray((optim.y_pred - optim.y_true)) # (bs, out_dim)
            d1 = np.multiply(d1, train_mask) # (bs, out_dim) with loss of non-train nodes set to zero
            
            self.grad = d1 @ self.W # (bs, out_dim)*(out_dim, in_dim) = (bs, in_dim)
            optim.out = self.grad
            
            dW = (d1.T @ self._X.T) / optim.bs  # (out_dim, bs)*(bs, in_dim) -> (out_dim, in_dim)
            db = d1.T.sum(axis=1, keepdims=True) / optim.bs # (out_dim, 1)
                    
            dW_wd = self.W * optim.wd / optim.bs # weight decay update
            
            if update:   
                self.W -= (dW + dW_wd) * optim.lr
                self.b -= db.reshape(self.b.shape) * optim.lr
            
            return dW + dW_wd, db.reshape(self.b.shape)
    def get_grads(inputs, layer, argname, labels, eps=1e-4, wd=0):
        cp = getattr(layer, argname).copy()
        cp_flat = np.asarray(cp).flatten()
        grads = np.zeros_like(cp_flat)
        n_parms = cp_flat.shape[0]
        for i, theta in enumerate(cp_flat):
            #print(f"Parm {argname}_{i}")
            theta_cp = theta
            
            # J(theta + eps)
            cp_flat[i] = theta + eps
            cp_tmp = cp_flat.reshape(cp.shape)
            predp = layer.forward(*inputs, **{argname: cp_tmp})
            wd_term = wd/2*(cp_flat**2).sum() / labels.shape[0]
            #print(wd_term)
            Jp = xent(predp, labels).mean() + wd_term
            
            # J(theta - eps)
            cp_flat[i] = theta - eps
            cp_tmp = cp_flat.reshape(cp.shape)
            predm = layer.forward(*inputs, **{argname: cp_tmp})
            wd_term = wd/2*(cp_flat**2).sum() / labels.shape[0]
            #print(wd_term)
            Jm = xent(predm, labels).mean() + wd_term
            
            # grad
            grads[i] = ((Jp - Jm) / (2*eps))
            
            # Back to normal
            cp_flat[i] = theta

        return grads.reshape(cp.shape)

    def get_gcn_grads(inputs, gcn, sm_layer, labels, eps=1e-4, wd=0):
        cp = gcn.W.copy()
        cp_flat = np.asarray(cp).flatten()
        grads = np.zeros_like(cp_flat)
        n_parms = cp_flat.shape[0]
        for i, theta in enumerate(cp_flat):
            theta_cp = theta
            
            # J(theta + eps)
            cp_flat[i] = theta + eps
            cp_tmp = cp_flat.reshape(cp.shape)
            pred = sm_layer.forward(gcn.forward(*inputs, W=cp_tmp))
            w2 = (cp_flat**2).sum()+(sm_layer.W.flatten()**2).sum()
            Jp = xent(pred, labels).mean() + wd/(2*labels.shape[0])*w2
            
            # J(theta - eps)
            cp_flat[i] = theta - eps
            cp_tmp = cp_flat.reshape(cp.shape)
            pred = sm_layer.forward(gcn.forward(*inputs, W=cp_tmp))
            w2 = (cp_flat**2).sum()+(sm_layer.W.flatten()**2).sum()
            Jm = xent(pred, labels).mean() + wd/(2*labels.shape[0])*w2
            
            # grad
            grads[i] = ((Jp - Jm) / (2*eps))
            
            # Back to normal
            cp_flat[i] = theta

        return grads.reshape(cp.shape)
    
    def get_gcn_input_grads(A_hat, X, gcn, sm_layer, labels, eps=1e-4):
        cp = X.copy()
        cp_flat = np.asarray(cp).flatten()
        grads = np.zeros_like(cp_flat)
        n_parms = cp_flat.shape[0]
        for i, x in enumerate(cp_flat):
            x_cp = x
            
            # J(theta + eps)
            cp_flat[i] = x + eps
            cp_tmp = cp_flat.reshape(cp.shape)
            pred = sm_layer.forward(gcn.forward(A_hat, cp_tmp))
            Jp = xent(pred, labels).mean()
            
            # J(theta - eps)
            cp_flat[i] = x - eps
            cp_tmp = cp_flat.reshape(cp.shape)
            pred = sm_layer.forward(gcn.forward(A_hat, cp_tmp))
            Jm = xent(pred, labels).mean()
            
            # grad
            grads[i] = ((Jp - Jm) / (2*eps))
            
            # Back to normal
            cp_flat[i] = x

        return grads.reshape(cp.shape)

    class GCNNetwork():
        def __init__(self, n_inputs, n_outputs, n_layers, hidden_sizes, activation, seed=0):
            self.n_inputs = n_inputs
            self.n_outputs = n_outputs
            self.n_layers = n_layers
            self.hidden_sizes = hidden_sizes
            self.activation = activation
            
            np.random.seed(seed)
            
            self.layers = list()
            # Input layer
            gcn_in = GCNLayer(n_inputs, hidden_sizes[0], activation, name='in')
            self.layers.append(gcn_in)
            
            # Hidden layers
            for layer in range(n_layers):
                gcn = GCNLayer(self.layers[-1].W.shape[0], hidden_sizes[layer], activation, name=f'h{layer}')
                self.layers.append(gcn)
                
            # Output layer
            sm_out = SoftmaxLayer(hidden_sizes[-1], n_outputs, name='sm')
            self.layers.append(sm_out)
            
        def __repr__(self):
            return '\n'.join([str(l) for l in self.layers])
        
        def embedding(self, A, X):
            # Loop through all GCN layers
            H = X
            for layer in self.layers[:-1]:
                H = layer.forward(A, H)
            return np.asarray(H)
        
        def forward(self, A, X):
            # GCN layers
            H = self.embedding(A, X)
            
            # Softmax
            p = self.layers[-1].forward(H)
            
            return np.asarray(p)
    
    def user_sort(input_id):
        df_user = []
        x0 = pos[input_id-1][0]
        y0 = pos[input_id-1][1]
        for i in range(len(pos)):
            d = math.sqrt((pos[i][0] - x0)**2 + (pos[i][1] - y0)**2)
            df_user.append(d)
            i = i + 1
        df_dsort = sorted(enumerate(df_user), key=itemgetter(1))
        return df_dsort

    def rcd(input_id, same_user, df_love):
        movie_name = pd.read_csv('ml-latest-small/movies.csv')
        #df_user = df_love[df_love['userId'] == input_id]
        df_same_love = df_love[df_love['userId'] == same_user]
        return df_same_love

    g, df_love = generate_g(50)

    # get color
    communities = greedy_modularity_communities(g)
    colors = np.zeros(g.number_of_nodes())
    for i, com in enumerate(communities):
        colors[list(com)] = i
    n_classes = np.unique(colors).shape[0]
    labels = np.eye(n_classes)[colors.astype(int)]
    
    # add label
    for v in g:
        g.nodes[v]['No'] = v + 1
    no_labels = nx.get_node_attributes(g,'No')
    no_labels

    #fig, ax = plt.subplots(figsize=(10,10))
    pos = nx.spring_layout(g, k=5/np.sqrt(g.number_of_nodes()))
    kwargs = {"cmap": 'gist_rainbow', "edge_color":'gray'}

    A = nx.to_numpy_matrix(g)
    A_mod = A + np.eye(g.number_of_nodes()) # add self-connections
    D_mod = np.zeros_like(A_mod)
    np.fill_diagonal(D_mod, np.asarray(A_mod.sum(axis=1)).flatten())
    D_mod_invroot = np.linalg.inv(sqrtm(D_mod))
    A_hat = D_mod_invroot @ A_mod @ D_mod_invroot
    X = np.eye(g.number_of_nodes())

    gcn1 = GCNLayer(g.number_of_nodes(), 2, activation=np.tanh, name='1')
    sm1 = SoftmaxLayer(2, n_classes, "SM")
    opt = GradDescentOptim(lr=0, wd=1.)

    gcn1_out = gcn1.forward(A_hat, X)
    opt(sm1.forward(gcn1_out), labels)
    dW_approx = get_grads((gcn1_out,), sm1, "W", labels, eps=1e-4, wd=opt.wd)
    db_approx = get_grads((gcn1_out,), sm1, "b", labels, eps=1e-4, wd=opt.wd)

    # Get gradients on Linear Softmax layer
    dW, db = sm1.backward(opt, update=False)

    assert norm_diff(dW, dW_approx) < 1e-7
    assert norm_diff(db, db_approx) < 1e-7

    dW2 = gcn1.backward(opt, update=False)
    dW2_approx = get_gcn_grads((A_hat, X), gcn1, sm1, labels, eps=1e-4, wd=opt.wd)
    assert norm_diff(dW2, dW2_approx) < 1e-7

    dX_approx = get_gcn_input_grads(A_hat, X, gcn1, sm1, labels, eps=1e-4)
    assert norm_diff(gcn1.grad/A_hat.shape[0], dX_approx) < 1e-7

    gcn_model = GCNNetwork(
        n_inputs=g.number_of_nodes(), 
        n_outputs=n_classes, 
        n_layers=2,
        hidden_sizes=[16, 2], 
        activation=np.tanh,
        seed=100,
    )

    y_pred = gcn_model.forward(A_hat, X)
    embed = gcn_model.embedding(A_hat, X)

    pos = {i: embed[i,:] for i in range(embed.shape[0])}

    train_nodes = np.array([0, 1, 8])
    test_nodes = np.array([i for i in range(labels.shape[0]) if i not in train_nodes])
    opt2 = GradDescentOptim(lr=2e-2, wd=2.5e-2)

    embeds = list()
    accs = list()
    train_losses = list()
    test_losses = list()

    loss_min = 1e6
    es_iters = 0
    es_steps = 50
    # lr_rate_ramp = 0 #-0.05
    # lr_ramp_steps = 1000

    for epoch in range(15000):
        
        y_pred = gcn_model.forward(A_hat, X)

        opt2(y_pred, labels, train_nodes)
        
    #     if ((epoch+1) % lr_ramp_steps) == 0:
    #         opt2.lr *= 1+lr_rate_ramp
    #         print(f"LR set to {opt2.lr:.4f}")

        for layer in reversed(gcn_model.layers):
            layer.backward(opt2, update=True)
            
        embeds.append(gcn_model.embedding(A_hat, X))
        # Accuracy for non-training nodes
        acc = (np.argmax(y_pred, axis=1) == np.argmax(labels, axis=1))[
            [i for i in range(labels.shape[0]) if i not in train_nodes]
        ]
        accs.append(acc.mean())
        
        loss = xent(y_pred, labels)
        loss_train = loss[train_nodes].mean()
        loss_test = loss[test_nodes].mean()
        
        train_losses.append(loss_train)
        test_losses.append(loss_test)
        
        if loss_test < loss_min:
            loss_min = loss_test
            es_iters = 0
        else:
            es_iters += 1
            
        if es_iters > es_steps:
            print("Early stopping!")
            break
        
        if epoch % 100 == 0:
            print(f"Epoch: {epoch+1}, Train Loss: {loss_train:.3f}, Test Loss: {loss_test:.3f}")
            
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    pos = {i: embeds[-1][i,:] for i in range(embeds[-1].shape[0])}
    N = 500
    snapshots = np.linspace(0, len(embeds)-1, N).astype(int)
    kwargs = {'cmap': 'gist_rainbow', 'edge_color': 'gray', }#'node_size': 55}
    
    input_id = int(request.form.get('wd1'))
    if input_id > 49:
        return {'Error':'Please enter a UserId between 1-49'}
    #df_dsort = user_sort(input_id)
    same_user = user_sort(input_id)[1][0]+1
    movie_name = pd.read_csv('ml-latest-small/movies.csv')
    df_rcd = rcd(input_id, same_user, df_love)
    df_rcd = df_rcd.merge(movie_name, on = 'movieId')

    out_json = {}
    movie_rcd = {}
    for i in range(df_rcd.shape[0]):
        movie_rcd[df_rcd.iloc[i]['title']] = df_rcd.iloc[i]['genres']
    out_json = {
        'Most similar User':{
            'Id': same_user,
            'Distance': user_sort(input_id)[1][1]
        },
        'Movie Recommended': movie_rcd}
    return out_json




if __name__ == '__main__':
    app.run(debug = True)






