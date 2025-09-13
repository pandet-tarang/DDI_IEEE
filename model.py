from datetime import datetime
import time
import argparse
from torch import optim
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import (GATConv, SAGPooling, LayerNorm, global_add_pool)
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from lr_scheduler import AdaptiveLR
from data_preprocessing import DrugDataset, DrugDataLoader, TOTAL_ATOM_FEATS
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

######################### Parameters ###################### 
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=TOTAL_ATOM_FEATS, help='num of input features')
parser.add_argument('--n_atom_hid', type=int, default=128, help='num of hidden features')
parser.add_argument('--rel_total', type=int, default=86, help='num of interaction types')
parser.add_argument('--lr', type=float, default=0.0012, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=50, help='num of epochs')
parser.add_argument('--kge_dim', type=int, default=64, help='dimension of interaction matrix')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')

parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])

args = parser.parse_args()
n_atom_feats = args.n_atom_feats
n_atom_hid = args.n_atom_hid
rel_total = args.rel_total
lr = args.lr
n_epochs = args.n_epochs
kge_dim = args.kge_dim
batch_size = args.batch_size

weight_decay = args.weight_decay
neg_samples = args.neg_samples
data_size_ratio = args.data_size_ratio
device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
print(args)
############################################################

###### Dataset
df_ddi_train = pd.read_csv('/content/drive/MyDrive/reserch_files/data/ddi_training.csv')
df_ddi_val = pd.read_csv('/content/drive/MyDrive/reserch_files/data/ddi_validation.csv')
df_ddi_test = pd.read_csv('/content/drive/MyDrive/reserch_files/data/ddi_test.csv')

train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
val_tup = [(h, t, r) for h, t, r in zip(df_ddi_val['d1'], df_ddi_val['d2'], df_ddi_val['type'])]
test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]

train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
val_data = DrugDataset(val_tup, ratio=data_size_ratio, disjoint_split=False)
test_data = DrugDataset(test_tup, disjoint_split=False)

print(f"Training with {len(train_data)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data_loader = DrugDataLoader(val_data, batch_size=batch_size * 3)
test_data_loader = DrugDataLoader(test_data, batch_size=batch_size * 3)


def do_compute(batch, device, training=True):
    '''
        *batch: (pos_tri, neg_tri)
        *pos/neg_tri: (batch_h, batch_t, batch_r)
    '''
    probas_pred, ground_truth = [], []
    pos_tri, neg_tri = batch

    pos_tri = [tensor.to(device=device) for tensor in pos_tri]
    p_score = model(pos_tri)
    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))

    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    n_score = model(neg_tri)
    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
    ground_truth.append(np.zeros(len(n_score)))

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)

    return p_score, n_score, probas_pred, ground_truth


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(np.int64)

    acc = metrics.accuracy_score(target, pred)
    auc_roc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)

    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    auc_prc = metrics.auc(r, p)

    return acc, auc_roc, auc_prc, precision, recall

class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features // 2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))

    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        attentions = e_scores

        return attentions


class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, heads, tails, rels, alpha_scores):
        rels = self.rel_emb(rels)
        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)
        rels = rels.view(-1, self.n_features, self.n_features)

        scores = heads @ rels @ tails.transpose(-2, -1)

        if alpha_scores is not None:
            scores = alpha_scores * scores
        scores = scores.sum(dim=(-2, -1))
        return scores

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"


class GAN_DDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params, dropout_rate=0.5):
        super().__init__()
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.n_blocks = len(blocks_params)

        self.initial_norm = LayerNorm(self.in_features)
        self.blocks = []
        self.net_norms = ModuleList()
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = GAN_DDI_Block(n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim, dropout_rate=dropout_rate)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads

        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

    def forward(self, triples):
        h_data, t_data, rels = triples

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)

        repr_h = []
        repr_t = []

        for i, block in enumerate(self.blocks):
            out1, out2 = block(h_data), block(t_data)

            h_data = out1[0]
            t_data = out2[0]
            r_h = out1[1]
            r_t = out2[1]

            repr_h.append(r_h)
            repr_t.append(r_t)

            h_data.x = F.elu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(self.net_norms[i](t_data.x, t_data.batch))

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)

        kge_heads = repr_h
        kge_tails = repr_t

        attentions = self.co_attention(kge_heads, kge_tails)
        scores = self.KGE(kge_heads, kge_tails, rels, attentions)

        return scores


class GAN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats, dropout_rate=0.5):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        self.conv = GATConv(in_features, head_out_feats, n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, data):
        data.x = self.conv(data.x, data.edge_index)
        att_x, att_edge_index, att_edge_attr, att_batch, att_perm, att_scores = self.readout(data.x, data.edge_index, batch=data.batch)
        att_x = self.dropout(att_x)
        global_graph_emb = global_add_pool(att_x, att_batch)
        return data, global_graph_emb


def train(model, train_data_loader, val_data_loader, loss_fn, optimizer, n_epochs, device):
    print('Starting training at', datetime.today())
    adaptive_lr = AdaptiveLR(optimizer, initial_lr=optimizer.param_groups[0]['lr'])
    
    for epoch in range(1, n_epochs + 1):
        start = time.time()
        train_loss = 0
        val_loss = 0
        train_probas_pred = []
        train_ground_truth = []
        val_probas_pred = []
        val_ground_truth = []

        # Training phase
        model.train()
        for batch in train_data_loader:
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device)
            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            loss, _, _ = loss_fn(p_score, n_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(p_score)
        train_loss /= len(train_data_loader.dataset)

        # Validation phase
        model.eval()
        with torch.no_grad():
            for batch in val_data_loader:
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device)
                val_probas_pred.append(probas_pred)
                val_ground_truth.append(ground_truth)
                loss, _, _ = loss_fn(p_score, n_score)
                val_loss += loss.item() * len(p_score)
            
            val_loss /= len(val_data_loader.dataset)
        
        # Compute metrics
        train_probas_pred = np.concatenate(train_probas_pred)
        train_ground_truth = np.concatenate(train_ground_truth)
        val_probas_pred = np.concatenate(val_probas_pred)
        val_ground_truth = np.concatenate(val_ground_truth)
        
        train_acc, train_auc_roc, train_auc_prc, train_precision, train_recall = do_compute_metrics(train_probas_pred, train_ground_truth)
        val_acc, val_auc_roc, val_auc_prc, val_precision, val_recall = do_compute_metrics(val_probas_pred, val_ground_truth)
        
        # Adjust learning rate
        adaptive_lr.step(val_loss)
        
        print(f'Epoch: {epoch} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f},'
              f' train_acc: {train_acc:.4f}, val_acc:{val_acc:.4f}')
        print(f'\ttrain_roc: {train_auc_roc:.4f}, val_roc: {val_auc_roc:.4f}, '
              f'train_auprc: {train_auc_prc:.4f}, val_auprc: {val_auc_prc:.4f}')
        print(f'\ttrain_precision: {train_precision:.4f}, train_recall: {train_recall:.4f}, '
              f'val_precision: {val_precision:.4f}, val_recall: {val_recall:.4f}')
        print(f'\tLearning Rate: {adaptive_lr.get_lr()}')

    return model, val_auc_roc


class SigmoidLoss(nn.Module):
    def __init__(self, adv_temperature=None):
        super().__init__()
        self.adv_temperature = adv_temperature

    def forward(self, p_scores, n_scores):
        if self.adv_temperature:
            weights = F.softmax(self.adv_temperature * n_scores, dim=-1).detach()
            n_scores = weights * n_scores
        p_loss = - F.logsigmoid(p_scores).mean()
        n_loss = - F.logsigmoid(-n_scores).mean()

        return (p_loss + n_loss) / 2, p_loss, n_loss


model = GAN_DDI(n_atom_feats, n_atom_hid, kge_dim, rel_total, heads_out_feat_params=[32, 32, 32, 32], blocks_params=[2, 2, 2, 2], dropout_rate=0.5)

loss = SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
model.to(device=device)

# Train the GAN_DDI model
trained_model, best_val_auc_roc = train(model, train_data_loader, val_data_loader, loss, optimizer, n_epochs, device)

# Function to extract features from the GAN_DDI model
def extract_features(model, data_loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch in data_loader:
            pos_tri, neg_tri = batch
            pos_tri = [tensor.to(device=device) for tensor in pos_tri]
            p_score = model(pos_tri)
            features.append(p_score.cpu().numpy())
            labels.append(np.ones(len(p_score)))
            
            neg_tri = [tensor.to(device=device) for tensor in neg_tri]
            n_score = model(neg_tri)
            features.append(n_score.cpu().numpy())
            labels.append(np.zeros(len(n_score)))
    
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features.reshape(-1, 1), labels

# Extract features from the trained GAN_DDI model
train_features, train_labels = extract_features(trained_model, train_data_loader, device)
val_features, val_labels = extract_features(trained_model, val_data_loader, device)
test_features, test_labels = extract_features(trained_model, test_data_loader, device)

# Train the Gradient Boosting model
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
gb_clf.fit(train_features, train_labels)

# Evaluate the combined model
val_predictions = gb_clf.predict_proba(val_features)[:, 1]
test_predictions = gb_clf.predict_proba(test_features)[:, 1]

val_acc, val_auc_roc, val_auc_prc, val_precision, val_recall = do_compute_metrics(val_predictions, val_labels)
test_acc, test_auc_roc, test_auc_prc, test_precision, test_recall = do_compute_metrics(test_predictions, test_labels)


#print(f"Validation AUC-ROC: {val_auc_roc:.4f}, AUC-PRC: {val_auc_prc:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
#print(f"Test AUC-ROC: {test_auc_roc:.4f}, AUC-PRC: {test_auc_prc:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")