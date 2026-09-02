# ================================================================
#  PolyGuard — Phase 1, Week 3
#  evaluate_model.py — Model Evaluation and Analysis
#  Status: Week 3 / 20
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
import os

os.makedirs('output/charts', exist_ok=True)

print("=" * 65)
print("  PolyGuard — Model Evaluation and Analysis")
print("  Phase 1 | Week 3 | Step 3 of 20")
print("=" * 65)

# ── STEP 1: RE-DEFINE MODEL ARCHITECTURE ──────────────────────
# Must match train_gnn.py exactly to load saved weights

class PolyGuardGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads, dropout):
        super(PolyGuardGNN, self).__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads,
                            dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim,
                            heads=heads, dropout=dropout, concat=True)
        self.gat3 = GATConv(hidden_dim * heads, out_dim,
                            heads=1, dropout=dropout, concat=False)
        self.comorbidity_proj = nn.Linear(out_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)
        self.edge_scorer = nn.Sequential(
            nn.Linear(out_dim * 2, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def encode(self, x, edge_index, comorbidity_context=None):
        x = F.elu(self.bn1(self.gat1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.gat2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat3(x, edge_index)
        if comorbidity_context is not None:
            x = x + self.comorbidity_proj(comorbidity_context)
        return x

    def decode(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        return self.edge_scorer(
            torch.cat([src, dst], dim=-1)
        ).squeeze(-1)

    def forward(self, x, edge_index, pred_edges,
                comorbidity_context=None):
        z     = self.encode(x, edge_index, comorbidity_context)
        score = self.decode(z, pred_edges)
        return score, z


# ── STEP 2: LOAD EVERYTHING ───────────────────────────────────

print("\n[STEP 1/6] Loading model and graph...")

graph = torch.load('output/polyguard_graph.pt', weights_only=False)

with open('output/drug_to_id.pkl', 'rb') as f:
    drug_to_id = pickle.load(f)
with open('output/id_to_drug.pkl', 'rb') as f:
    id_to_drug = pickle.load(f)
with open('output/twosides_lookup.pkl', 'rb') as f:
    twosides_lookup = pickle.load(f)

model = PolyGuardGNN(
    in_dim=128, hidden_dim=256,
    out_dim=128, heads=4, dropout=0.3
)
model.load_state_dict(
    torch.load('output/polyguard_model_best.pt', weights_only=True)
)
model.eval()

print(f"  Model loaded from   : output/polyguard_model_best.pt")
print(f"  Graph nodes         : {graph.num_nodes:,}")
print(f"  Graph edges         : {graph.num_edges:,}")
print(f"  Drug vocabulary     : {len(drug_to_id):,}")

# ── STEP 3: REBUILD TEST SPLIT ────────────────────────────────

print("\n[STEP 2/6] Rebuilding test split...")

torch.manual_seed(42)
np.random.seed(42)

edge_index = graph.edge_index
edge_attr  = graph.edge_attr
num_edges  = edge_index.shape[1]

perm       = torch.randperm(num_edges,
             generator=torch.Generator().manual_seed(42))
edge_index_s = edge_index[:, perm]
edge_attr_s  = edge_attr[perm]

train_end  = int(num_edges * 0.80)
val_end    = int(num_edges * 0.90)

train_edges = edge_index_s[:, :train_end]
test_edges  = edge_index_s[:, val_end:]
test_attr   = edge_attr_s[val_end:]

print(f"  Test edges          : {test_edges.shape[1]:,}")

# ── STEP 4: FULL TEST EVALUATION ──────────────────────────────

print("\n[STEP 3/6] Running full test evaluation...")

with torch.no_grad():
    neg_test = negative_sampling(
        edge_index=test_edges,
        num_nodes=graph.num_nodes,
        num_neg_samples=test_edges.shape[1]
    )
    test_combined = torch.cat([test_edges, neg_test], dim=1)
    test_labels   = torch.cat([
        torch.ones(test_edges.shape[1]),
        torch.zeros(neg_test.shape[1])
    ])
    test_attr_neg = torch.cat([
        test_attr,
        torch.zeros(neg_test.shape[1], dtype=torch.long)
    ])

    scores, _ = model(graph.x, train_edges, test_combined)
    probs     = torch.sigmoid(scores).numpy()
    preds     = (probs > 0.5).astype(int)
    true      = test_labels.numpy().astype(int)
    attrs     = test_attr_neg.numpy()

auroc     = roc_auc_score(true, probs)
f1        = f1_score(true, preds, zero_division=0)
precision = precision_score(true, preds, zero_division=0)
recall    = recall_score(true, preds, zero_division=0)
cm        = confusion_matrix(true, preds)

print(f"\n  {'Metric':<20} {'Value':>10}")
print(f"  {'-'*30}")
print(f"  {'AUROC':<20} {auroc:>10.4f}")
print(f"  {'F1 Score':<20} {f1:>10.4f}")
print(f"  {'Precision':<20} {precision:>10.4f}")
print(f"  {'Recall':<20} {recall:>10.4f}")
print(f"\n  Confusion Matrix:")
print(f"  TN={cm[0][0]:,}  FP={cm[0][1]:,}")
print(f"  FN={cm[1][0]:,}  TP={cm[1][1]:,}")

# Severity-wise breakdown
print(f"\n  Severity-wise breakdown:")
for sev_code, sev_name in [(2, 'Severe'), (1, 'Moderate'), (0, 'Mild')]:
    mask = attrs == sev_code
    if mask.sum() > 10:
        sev_auroc = roc_auc_score(true[mask], probs[mask])
        sev_f1    = f1_score(true[mask], preds[mask], zero_division=0)
        print(f"  {sev_name:<12} AUROC={sev_auroc:.4f}  F1={sev_f1:.4f}"
              f"  n={mask.sum()}")

# ── STEP 5: GENERATE CHARTS ───────────────────────────────────

print("\n[STEP 4/6] Generating evaluation charts...")

# Chart 1 — Training history
with open('output/training_history.json', 'r') as f:
    history = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('PolyGuard CC-DGNN — Training History', fontsize=14)

epochs_logged = list(range(1, len(history['loss']) + 1))

axes[0].plot(epochs_logged, history['loss'], 'b-o', markersize=3)
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Checkpoint (every 5 epochs)')
axes[0].set_ylabel('Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_logged, history['val_auroc'],
             'g-o', markersize=3, label='Val AUROC')
axes[1].plot(epochs_logged, history['val_f1'],
             'r-o', markersize=3, label='Val F1')
axes[1].axhline(y=0.90, color='gray', linestyle='--',
                alpha=0.7, label='Target 0.90')
axes[1].set_title('Validation Metrics')
axes[1].set_xlabel('Checkpoint (every 5 epochs)')
axes[1].set_ylabel('Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/charts/training_history.png', dpi=150)
plt.close()
print(f"  Saved: output/charts/training_history.png")

# Chart 2 — ROC Curve
fpr, tpr, _ = roc_curve(true, probs)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, 'b-', linewidth=2,
        label=f'CC-DGNN (AUROC = {auroc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random baseline')
ax.fill_between(fpr, tpr, alpha=0.1, color='blue')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('PolyGuard — ROC Curve')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/charts/roc_curve.png', dpi=150)
plt.close()
print(f"  Saved: output/charts/roc_curve.png")

# Chart 3 — Precision-Recall Curve
prec_curve, rec_curve, _ = precision_recall_curve(true, probs)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec_curve, prec_curve, 'g-', linewidth=2)
ax.fill_between(rec_curve, prec_curve, alpha=0.1, color='green')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('PolyGuard — Precision-Recall Curve')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/charts/precision_recall.png', dpi=150)
plt.close()
print(f"  Saved: output/charts/precision_recall.png")

# Chart 4 — Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['No Interaction', 'Interaction'])
ax.set_yticklabels(['No Interaction', 'Interaction'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('PolyGuard — Confusion Matrix')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i][j]:,}',
                ha='center', va='center',
                color='white' if cm[i][j] > cm.max()/2 else 'black',
                fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('output/charts/confusion_matrix.png', dpi=150)
plt.close()
print(f"  Saved: output/charts/confusion_matrix.png")

# ── STEP 6: LIVE DRUG PAIR PREDICTION ────────────────────────

print("\n[STEP 5/6] Testing live drug pair predictions...")

def predict_interaction(drug1_name, drug2_name):
    d1 = drug1_name.strip().lower()
    d2 = drug2_name.strip().lower()

    if d1 not in drug_to_id:
        return f"  '{drug1_name}' not found in drug vocabulary"
    if d2 not in drug_to_id:
        return f"  '{drug2_name}' not found in drug vocabulary"

    id1 = drug_to_id[d1]
    id2 = drug_to_id[d2]

    pred_edge = torch.tensor([[id1], [id2]], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        score, _ = model(graph.x, train_edges, pred_edge)
        prob     = torch.sigmoid(score).item()

    if prob >= 0.80:
        risk  = 'HIGH RISK'
        badge = 'ALERT'
    elif prob >= 0.50:
        risk  = 'MODERATE RISK'
        badge = 'CAUTION'
    else:
        risk  = 'LOW RISK'
        badge = 'SAFE'

    side_effect = twosides_lookup.get((d1, d2),
                  twosides_lookup.get((d2, d1), 'Not in TWOSIDES'))

    result = (
        f"\n  Drug 1          : {drug1_name.title()}\n"
        f"  Drug 2          : {drug2_name.title()}\n"
        f"  Risk Score      : {prob:.4f}\n"
        f"  Risk Level      : {risk}\n"
        f"  Status          : [{badge}]\n"
        f"  Side Effect     : {side_effect}"
    )
    return result

# Test with known dangerous pairs
test_pairs = [
    ("Warfarin",    "Aspirin"),
    ("Metformin",   "Alcohol"),
    ("Sertraline",  "Tramadol"),
    ("Amoxicillin", "Methotrexate"),
    ("Lisinopril",  "Potassium"),
]

print(f"\n  Testing {len(test_pairs)} known drug pairs:\n")
print(f"  {'─' * 55}")

for d1, d2 in test_pairs:
    print(predict_interaction(d1, d2))
    print(f"  {'─' * 55}")

# ── STEP 7: SAVE EVALUATION REPORT ───────────────────────────

print("\n[STEP 6/6] Saving evaluation report...")

report = {
    'test_auroc'    : round(auroc, 4),
    'test_f1'       : round(f1, 4),
    'test_precision': round(precision, 4),
    'test_recall'   : round(recall, 4),
    'confusion_matrix': cm.tolist(),
    'best_val_auroc': max(history['val_auroc']),
}

with open('output/evaluation_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"  output/evaluation_report.json    — full metrics")
print(f"  output/charts/training_history.png")
print(f"  output/charts/roc_curve.png")
print(f"  output/charts/precision_recall.png")
print(f"  output/charts/confusion_matrix.png")
print(f"\n  Phase 1 | Week 3 COMPLETE")
print(f"  NEXT STEP: Week 4 — bayesian_layer.py (alert fatigue reducer)")
print("=" * 65)