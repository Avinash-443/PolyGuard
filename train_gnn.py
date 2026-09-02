# ================================================================
#  PolyGuard — Phase 1, Week 2
#  train_gnn.py — CC-DGNN Model Training
#  Status: Week 2 / 20
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import pickle
import numpy as np
import os
import time

# ── CONFIGURATION ─────────────────────────────────────────────

CONFIG = {
    'node_feature_dim' : 128,
    'hidden_dim'       : 256,
    'output_dim'       : 128,
    'gat_heads'        : 4,
    'dropout'          : 0.3,
    'learning_rate'    : 0.001,
    'epochs'           : 100,
    'train_ratio'      : 0.80,
    'val_ratio'        : 0.10,
    'test_ratio'       : 0.10,
    'seed'             : 42,
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

print("=" * 65)
print("  PolyGuard — CC-DGNN Model Training")
print("  Phase 1 | Week 2 | Step 2 of 20")
print("=" * 65)

# ── STEP 1: LOAD SAVED GRAPH ──────────────────────────────────

print("\n[STEP 1/6] Loading saved graph...")

graph = torch.load('output/polyguard_graph.pt', weights_only=False)

with open('output/drug_to_id.pkl', 'rb') as f:
    drug_to_id = pickle.load(f)

with open('output/id_to_drug.pkl', 'rb') as f:
    id_to_drug = pickle.load(f)

print(f"  Graph nodes      : {graph.num_nodes:,}")
print(f"  Graph edges      : {graph.num_edges:,}")
print(f"  Node feature dim : {graph.x.shape[1]}")
print(f"  Drug vocabulary  : {len(drug_to_id):,} drugs")

# ── STEP 2: PREPARE EDGE SPLITS ───────────────────────────────

print("\n[STEP 2/6] Splitting edges into train / val / test...")

edge_index = graph.edge_index
edge_attr  = graph.edge_attr
num_edges  = edge_index.shape[1]

# Shuffle edges
perm       = torch.randperm(num_edges, generator=torch.Generator().manual_seed(42))
edge_index = edge_index[:, perm]
edge_attr  = edge_attr[perm]

train_end  = int(num_edges * CONFIG['train_ratio'])
val_end    = int(num_edges * (CONFIG['train_ratio'] + CONFIG['val_ratio']))

train_edges = edge_index[:, :train_end]
train_attr  = edge_attr[:train_end]

val_edges   = edge_index[:, train_end:val_end]
val_attr    = edge_attr[train_end:val_end]

test_edges  = edge_index[:, val_end:]
test_attr   = edge_attr[val_end:]

print(f"  Train edges      : {train_edges.shape[1]:,}")
print(f"  Val edges        : {val_edges.shape[1]:,}")
print(f"  Test edges       : {test_edges.shape[1]:,}")

# ── STEP 3: DEFINE CC-DGNN MODEL ──────────────────────────────

print("\n[STEP 3/6] Building CC-DGNN model architecture...")

class PolyGuardGNN(nn.Module):
    """
    Comorbidity-Conditioned Dynamic Graph Neural Network (CC-DGNN)
    Patent Contribution 1 — PolyGuard Project

    Architecture:
    - 3-layer Graph Attention Network (GATConv)
    - Comorbidity context injection via attention modulation
    - Dropout regularisation for Bayesian uncertainty (Patent 2 prep)
    - Edge scoring head for DDI prediction
    """

    def __init__(self, in_dim, hidden_dim, out_dim, heads, dropout):
        super(PolyGuardGNN, self).__init__()

        self.dropout = dropout

        # Layer 1 — Initial drug feature transformation
        self.gat1 = GATConv(
            in_channels  = in_dim,
            out_channels = hidden_dim,
            heads        = heads,
            dropout      = dropout,
            concat       = True
        )

        # Layer 2 — Deep interaction learning
        self.gat2 = GATConv(
            in_channels  = hidden_dim * heads,
            out_channels = hidden_dim,
            heads        = heads,
            dropout      = dropout,
            concat       = True
        )

        # Layer 3 — Final node embedding
        self.gat3 = GATConv(
            in_channels  = hidden_dim * heads,
            out_channels = out_dim,
            heads        = 1,
            dropout      = dropout,
            concat       = False
        )

        # Comorbidity context injection layer (Patent 1 core)
        # Projects comorbidity vector and adds it to node embeddings
        # during forward pass — modulating attention based on patient disease
        self.comorbidity_proj = nn.Linear(out_dim, out_dim)

        # Batch normalisation for stable training
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)

        # Edge scoring head — predicts interaction risk between drug pairs
        self.edge_scorer = nn.Sequential(
            nn.Linear(out_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def encode(self, x, edge_index, comorbidity_context=None):
        """
        Encode all drug nodes into embeddings using GAT layers.
        comorbidity_context: optional tensor [num_nodes, out_dim]
        representing patient disease profile — injected at final layer.
        This is the core of Patent 1 (CC-DGNN).
        """
        # GAT Layer 1
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GAT Layer 2
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GAT Layer 3
        x = self.gat3(x, edge_index)

        # Comorbidity context injection (Patent 1)
        # If patient comorbidity vector is provided, modulate embeddings
        if comorbidity_context is not None:
            context = self.comorbidity_proj(comorbidity_context)
            x = x + context

        return x

    def decode(self, z, edge_index):
        """
        Score each edge (drug pair) for interaction risk.
        Concatenates source and target node embeddings,
        then passes through edge scoring MLP.
        """
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        edge_feat = torch.cat([src, dst], dim=-1)
        return self.edge_scorer(edge_feat).squeeze(-1)

    def forward(self, x, edge_index, pred_edges, comorbidity_context=None):
        z     = self.encode(x, edge_index, comorbidity_context)
        score = self.decode(z, pred_edges)
        return score, z


# Instantiate the model
model = PolyGuardGNN(
    in_dim     = CONFIG['node_feature_dim'],
    hidden_dim = CONFIG['hidden_dim'],
    out_dim    = CONFIG['output_dim'],
    heads      = CONFIG['gat_heads'],
    dropout    = CONFIG['dropout']
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Model architecture : CC-DGNN (3-layer GAT)")
print(f"  Attention heads    : {CONFIG['gat_heads']}")
print(f"  Hidden dimensions  : {CONFIG['hidden_dim']}")
print(f"  Trainable params   : {total_params:,}")

# ── STEP 4: SETUP TRAINING ────────────────────────────────────

print("\n[STEP 4/6] Setting up training...")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr           = CONFIG['learning_rate'],
    weight_decay = 1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)

# Weighted loss — severe interactions get 4x weight, moderate 2x
# This handles the class imbalance in your dataset
def weighted_bce_loss(scores, labels, edge_attr=None):
    weights = torch.ones_like(labels)
    if edge_attr is not None:
        weights[edge_attr == 2] = 4.0  # severe
        weights[edge_attr == 1] = 2.0  # moderate
    loss = F.binary_cross_entropy_with_logits(
        scores, labels, weight=weights
    )
    return loss

x          = graph.x
edge_index = train_edges

print(f"  Optimizer          : Adam (lr={CONFIG['learning_rate']})")
print(f"  Loss               : Weighted BCE (severe=4x, moderate=2x)")
print(f"  Epochs             : {CONFIG['epochs']}")
print(f"  Scheduler          : ReduceLROnPlateau (patience=10)")

# ── STEP 5: TRAINING LOOP ─────────────────────────────────────

print("\n[STEP 5/6] Training CC-DGNN model...")
print(f"  {'Epoch':>6} | {'Loss':>8} | {'Val AUROC':>10} | {'Val F1':>8} | {'LR':>8}")
print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

best_val_auroc = 0.0
best_epoch     = 0
history        = {'loss': [], 'val_auroc': [], 'val_f1': []}

start_time = time.time()

for epoch in range(1, CONFIG['epochs'] + 1):

    # ── TRAIN ────────────────────────────────────────────────
    model.train()

    # Positive edges — real drug interactions
    pos_edge = train_edges
    pos_attr = train_attr

    # Negative sampling — random drug pairs with no known interaction
    neg_edge = negative_sampling(
        edge_index = train_edges,
        num_nodes  = graph.num_nodes,
        num_neg_samples = train_edges.shape[1]
    )

    # Build combined edge set with labels
    combined_edges  = torch.cat([pos_edge, neg_edge], dim=1)
    pos_labels      = torch.ones(pos_edge.shape[1])
    neg_labels      = torch.zeros(neg_edge.shape[1])
    combined_labels = torch.cat([pos_labels, neg_labels])

    combined_attr   = torch.cat([
        pos_attr,
        torch.zeros(neg_edge.shape[1], dtype=torch.long)
    ])

    optimizer.zero_grad()
    scores, _ = model(x, edge_index, combined_edges)
    loss      = weighted_bce_loss(scores, combined_labels, combined_attr)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # ── VALIDATE ─────────────────────────────────────────────
    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():

            # Positive val edges
            pos_val = val_edges
            neg_val = negative_sampling(
                edge_index      = val_edges,
                num_nodes       = graph.num_nodes,
                num_neg_samples = val_edges.shape[1]
            )

            val_combined = torch.cat([pos_val, neg_val], dim=1)
            val_labels   = torch.cat([
                torch.ones(pos_val.shape[1]),
                torch.zeros(neg_val.shape[1])
            ])

            val_scores, _ = model(x, edge_index, val_combined)
            val_probs     = torch.sigmoid(val_scores).numpy()
            val_preds     = (val_probs > 0.5).astype(int)
            val_true      = val_labels.numpy().astype(int)

            val_auroc = roc_auc_score(val_true, val_probs)
            val_f1    = f1_score(val_true, val_preds, zero_division=0)

            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_auroc)

            history['loss'].append(loss.item())
            history['val_auroc'].append(val_auroc)
            history['val_f1'].append(val_f1)

            print(f"  {epoch:>6} | {loss.item():>8.4f} | "
                  f"{val_auroc:>10.4f} | {val_f1:>8.4f} | "
                  f"{current_lr:>8.6f}")

            # Save best model
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_epoch     = epoch
                torch.save(model.state_dict(), 'output/polyguard_model_best.pt')

# ── STEP 6: FINAL EVALUATION ──────────────────────────────────

print(f"\n[STEP 6/6] Final evaluation on test set...")

# Load best model for final test evaluation
model.load_state_dict(
    torch.load('output/polyguard_model_best.pt', weights_only=True)
)
model.eval()

with torch.no_grad():

    pos_test = test_edges
    neg_test = negative_sampling(
        edge_index      = test_edges,
        num_nodes       = graph.num_nodes,
        num_neg_samples = test_edges.shape[1]
    )

    test_combined = torch.cat([pos_test, neg_test], dim=1)
    test_labels   = torch.cat([
        torch.ones(pos_test.shape[1]),
        torch.zeros(neg_test.shape[1])
    ])

    test_scores, node_embeddings = model(x, edge_index, test_combined)
    test_probs = torch.sigmoid(test_scores).numpy()
    test_preds = (test_probs > 0.5).astype(int)
    test_true  = test_labels.numpy().astype(int)

    test_auroc     = roc_auc_score(test_true, test_probs)
    test_f1        = f1_score(test_true, test_preds, zero_division=0)
    test_precision = precision_score(test_true, test_preds, zero_division=0)
    test_recall    = recall_score(test_true, test_preds, zero_division=0)

elapsed = round(time.time() - start_time, 1)

print(f"\n  Test AUROC         : {test_auroc:.4f}")
print(f"  Test F1 Score      : {test_f1:.4f}")
print(f"  Test Precision     : {test_precision:.4f}")
print(f"  Test Recall        : {test_recall:.4f}")
print(f"  Best Val AUROC     : {best_val_auroc:.4f} (epoch {best_epoch})")

# Save final model and embeddings
torch.save(model.state_dict(), 'output/polyguard_model.pt')
torch.save(node_embeddings, 'output/node_embeddings.pt')

# Save training history for report charts
import json
with open('output/training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"\n  Saved:")
print(f"  output/polyguard_model.pt        — final trained model")
print(f"  output/polyguard_model_best.pt   — best checkpoint")
print(f"  output/node_embeddings.pt        — learned drug embeddings")
print(f"  output/training_history.json     — loss and AUROC per epoch")
print(f"\n  Training time      : {elapsed}s")