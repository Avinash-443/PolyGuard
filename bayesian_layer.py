# ================================================================
#  PolyGuard — Phase 1, Week 4
#  bayesian_layer.py — Bayesian Uncertainty Alert Fatigue Reducer
#  Patent Contribution 2
#  Status: Week 4 / 20
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)
import numpy as np
import pickle
import json
import os

os.makedirs('output/bayesian', exist_ok=True)

print("=" * 65)
print("  PolyGuard — Bayesian Alert Fatigue Reducer")
print("  Phase 1 | Week 4 | Step 4 of 20")
print("  Patent Contribution 2")
print("=" * 65)

# ── STEP 1: RE-DEFINE MODEL WITH MC DROPOUT ───────────────────
#
# Key difference from train_gnn.py:
# In standard inference, dropout is DISABLED (model.eval())
# In Bayesian / MC Dropout inference, dropout stays ENABLED
# even during prediction — this creates slightly different
# outputs each time we run the same input through the model.
# Running it N times gives us a distribution of predictions.
# The spread of that distribution = uncertainty.
# Narrow spread = model is confident = show alert
# Wide spread   = model is uncertain = suppress alert
#
# This is Patent 2: confidence-calibrated alert suppression.

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


# ── STEP 2: LOAD MODEL AND DATA ───────────────────────────────

print("\n[STEP 1/6] Loading model and data...")

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

print(f"  Model loaded        : output/polyguard_model_best.pt")
print(f"  Graph nodes         : {graph.num_nodes:,}")
print(f"  Graph edges         : {graph.num_edges:,}")

# Rebuild train/test split (same seed as train_gnn.py)
torch.manual_seed(42)
np.random.seed(42)

edge_index  = graph.edge_index
edge_attr   = graph.edge_attr
num_edges   = edge_index.shape[1]
perm        = torch.randperm(
    num_edges,
    generator=torch.Generator().manual_seed(42)
)
edge_index_s = edge_index[:, perm]
edge_attr_s  = edge_attr[perm]

train_end    = int(num_edges * 0.80)
val_end      = int(num_edges * 0.90)
train_edges  = edge_index_s[:, :train_end]
test_edges   = edge_index_s[:, val_end:]
test_attr    = edge_attr_s[val_end:]

print(f"  Test edges          : {test_edges.shape[1]:,}")

# ── STEP 3: MC DROPOUT INFERENCE FUNCTION ─────────────────────

print("\n[STEP 2/6] Defining MC Dropout inference...")

def mc_dropout_predict(model, x, train_edges, pred_edges,
                       n_passes=30):
    """
    Monte Carlo Dropout Inference — Patent 2 core mechanism.

    Instead of one forward pass with dropout disabled,
    we run N forward passes with dropout ENABLED.
    Each pass gives a slightly different prediction.

    From N predictions we compute:
    - mean_prob   : average risk score (the prediction)
    - uncertainty : std deviation across passes
                    high std = model is uncertain = suppress alert
                    low std  = model is confident = show alert

    n_passes=30 gives stable uncertainty estimates.
    """

    # CRITICAL: keep model in TRAIN mode so dropout stays active
    model.train()

    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            scores, _ = model(x, train_edges, pred_edges)
            probs     = torch.sigmoid(scores)
            all_probs.append(probs.unsqueeze(0))

    # Stack: shape [n_passes, num_edges]
    all_probs   = torch.cat(all_probs, dim=0)

    mean_prob   = all_probs.mean(dim=0)   # average prediction
    uncertainty = all_probs.std(dim=0)    # spread = uncertainty

    return mean_prob.numpy(), uncertainty.numpy()


# ── STEP 4: BUILD TEST SET AND RUN MC DROPOUT ─────────────────

print("\n[STEP 3/6] Running MC Dropout on test set (30 passes)...")
print("  This takes a few minutes — running 30 forward passes...")

neg_test = negative_sampling(
    edge_index      = test_edges,
    num_nodes       = graph.num_nodes,
    num_neg_samples = test_edges.shape[1]
)

test_combined = torch.cat([test_edges, neg_test], dim=1)
test_labels   = torch.cat([
    torch.ones(test_edges.shape[1]),
    torch.zeros(neg_test.shape[1])
]).numpy()

mean_probs, uncertainties = mc_dropout_predict(
    model, graph.x, train_edges, test_combined, n_passes=30
)

print(f"  MC Dropout complete")
print(f"  Mean probability range  : "
      f"{mean_probs.min():.4f} — {mean_probs.max():.4f}")
print(f"  Uncertainty range       : "
      f"{uncertainties.min():.4f} — {uncertainties.max():.4f}")
print(f"  Mean uncertainty        : {uncertainties.mean():.4f}")

# ── STEP 5: CONFIDENCE-CALIBRATED ALERT FILTERING ─────────────
#
# An alert is shown to the doctor ONLY when:
# 1. mean_prob   > RISK_THRESHOLD    (model says it is risky)
# 2. uncertainty < CONF_THRESHOLD    (model is confident)
#
# If uncertainty is HIGH — model is unsure — alert is suppressed.
# Doctor still sees a soft warning: "Uncertain — verify manually"
# This is the patent mechanism: dual-threshold alert gate.

print("\n[STEP 4/6] Applying confidence-calibrated alert filter...")

RISK_THRESHOLD = 0.50   # minimum risk score to consider
CONF_THRESHOLD = 0.15   # maximum uncertainty to show alert

# Standard predictions (no Bayesian filter)
standard_preds = (mean_probs > RISK_THRESHOLD).astype(int)

# Bayesian-filtered predictions
# Only show alert if BOTH conditions met
bayesian_preds = (
    (mean_probs   >  RISK_THRESHOLD) &
    (uncertainties < CONF_THRESHOLD)
).astype(int)

true_labels = test_labels.astype(int)

# Compute metrics — standard vs Bayesian
std_precision = precision_score(true_labels, standard_preds,
                                zero_division=0)
std_recall    = recall_score(true_labels, standard_preds,
                             zero_division=0)
std_f1        = f1_score(true_labels, standard_preds,
                         zero_division=0)
std_alerts    = standard_preds.sum()

bay_precision = precision_score(true_labels, bayesian_preds,
                                zero_division=0)
bay_recall    = recall_score(true_labels, bayesian_preds,
                             zero_division=0)
bay_f1        = f1_score(true_labels, bayesian_preds,
                         zero_division=0)
bay_alerts    = bayesian_preds.sum()

# Alert reduction = fewer false alarms shown to doctor
total_alerts    = len(standard_preds)
alerts_reduced  = int(std_alerts - bay_alerts)
reduction_pct   = (alerts_reduced / std_alerts * 100
                   if std_alerts > 0 else 0)

# False positives caught — alerts that would have been wrong
std_fp  = ((standard_preds == 1) & (true_labels == 0)).sum()
bay_fp  = ((bayesian_preds == 1) & (true_labels == 0)).sum()
fp_saved = int(std_fp - bay_fp)

print(f"\n  {'Metric':<25} {'Standard':>12} {'Bayesian':>12}")
print(f"  {'─'*49}")
print(f"  {'Precision':<25} {std_precision:>12.4f} "
      f"{bay_precision:>12.4f}")
print(f"  {'Recall':<25} {std_recall:>12.4f} "
      f"{bay_recall:>12.4f}")
print(f"  {'F1 Score':<25} {std_f1:>12.4f} "
      f"{bay_f1:>12.4f}")
print(f"  {'Total Alerts Fired':<25} {std_alerts:>12,} "
      f"{bay_alerts:>12,}")
print(f"  {'False Positives':<25} {std_fp:>12,} "
      f"{bay_fp:>12,}")
print(f"\n  Alert reduction         : {alerts_reduced:,} "
      f"fewer alerts ({reduction_pct:.1f}%)")
print(f"  False positives saved   : {fp_saved:,} "
      f"wrong alerts suppressed")

# ── STEP 6: UNCERTAINTY THRESHOLD SWEEP ───────────────────────
#
# Find the best uncertainty threshold — the one that maximises
# precision improvement while keeping recall above 0.90
# This gives you the optimal operating point for your system

print("\n[STEP 5/6] Sweeping uncertainty thresholds...")

thresholds = np.arange(0.05, 0.31, 0.01)
results    = []

print(f"\n  {'Threshold':>10} | {'Precision':>10} | "
      f"{'Recall':>8} | {'F1':>8} | {'Alerts':>10} | {'FP Saved':>10}")
print(f"  {'─'*65}")

best_threshold = CONF_THRESHOLD
best_f1        = 0.0

for thresh in thresholds:
    preds = (
        (mean_probs   >  RISK_THRESHOLD) &
        (uncertainties < thresh)
    ).astype(int)

    p  = precision_score(true_labels, preds, zero_division=0)
    r  = recall_score(true_labels, preds, zero_division=0)
    f  = f1_score(true_labels, preds, zero_division=0)
    al = preds.sum()
    fp = ((preds == 1) & (true_labels == 0)).sum()
    fp_s = int(std_fp - fp)

    results.append({
        'threshold' : round(float(thresh), 2),
        'precision' : round(p, 4),
        'recall'    : round(r, 4),
        'f1'        : round(f, 4),
        'alerts'    : int(al),
        'fp_saved'  : fp_s
    })

    # Best threshold = highest F1 while recall stays above 0.90
    if f > best_f1 and r >= 0.90:
        best_f1        = f
        best_threshold = round(float(thresh), 2)

    marker = ' <-- best' if round(float(thresh), 2) == best_threshold else ''
    print(f"  {thresh:>10.2f} | {p:>10.4f} | "
          f"{r:>8.4f} | {f:>8.4f} | {al:>10,} | {fp_s:>10,}{marker}")

print(f"\n  Optimal uncertainty threshold : {best_threshold}")
print(f"  (Highest F1 while recall ≥ 0.90)")

# ── STEP 7: LIVE DRUG PAIR WITH UNCERTAINTY ────────────────────

print("\n[STEP 6/6] Testing live drug pairs with uncertainty scores...")

def predict_with_uncertainty(drug1_name, drug2_name, n_passes=30):
    d1 = drug1_name.strip().lower()
    d2 = drug2_name.strip().lower()

    if d1 not in drug_to_id:
        return f"  '{drug1_name}' not found in vocabulary"
    if d2 not in drug_to_id:
        return f"  '{drug2_name}' not found in vocabulary"

    id1 = drug_to_id[d1]
    id2 = drug_to_id[d2]

    pred_edge = torch.tensor([[id1], [id2]], dtype=torch.long)

    mean_p, uncert = mc_dropout_predict(
        model, graph.x, train_edges, pred_edge, n_passes=n_passes
    )
    mean_p  = float(mean_p[0])
    uncert  = float(uncert[0])

    # Apply dual-threshold gate (Patent 2 mechanism)
    is_risky     = mean_p   > RISK_THRESHOLD
    is_confident = uncert   < best_threshold

    if is_risky and is_confident:
        status = 'ALERT — High confidence interaction detected'
        badge  = 'RED'
    elif is_risky and not is_confident:
        status = 'UNCERTAIN — Interaction possible, verify manually'
        badge  = 'YELLOW'
    else:
        status = 'SAFE — No significant interaction detected'
        badge  = 'GREEN'

    side_effect = twosides_lookup.get(
        (d1, d2), twosides_lookup.get((d2, d1), 'Not in TWOSIDES')
    )

    return (
        f"\n  Drug 1          : {drug1_name.title()}\n"
        f"  Drug 2          : {drug2_name.title()}\n"
        f"  Risk Score      : {mean_p:.4f}\n"
        f"  Uncertainty     : {uncert:.4f}\n"
        f"  Confident?      : {'Yes' if is_confident else 'No'}\n"
        f"  Alert Badge     : [{badge}]\n"
        f"  Status          : {status}\n"
        f"  Side Effect     : {side_effect}"
    )

test_pairs = [
    ("Warfarin",     "Aspirin"),
    ("Sertraline",   "Tramadol"),
    ("Metformin",    "Alcohol"),
    ("Amoxicillin",  "Methotrexate"),
    ("Lisinopril",   "Potassium"),
]

print(f"\n  {'─' * 57}")
for d1, d2 in test_pairs:
    print(predict_with_uncertainty(d1, d2))
    print(f"  {'─' * 57}")

# ── SAVE RESULTS ──────────────────────────────────────────────

bayesian_report = {
    'optimal_uncertainty_threshold' : best_threshold,
    'risk_threshold'                : RISK_THRESHOLD,
    'n_mc_passes'                   : 30,
    'standard': {
        'precision' : round(std_precision, 4),
        'recall'    : round(std_recall, 4),
        'f1'        : round(std_f1, 4),
        'alerts'    : int(std_alerts),
        'false_positives': int(std_fp),
    },
    'bayesian': {
        'precision' : round(bay_precision, 4),
        'recall'    : round(bay_recall, 4),
        'f1'        : round(bay_f1, 4),
        'alerts'    : int(bay_alerts),
        'false_positives': int(bay_fp),
    },
    'improvement': {
        'alerts_reduced'  : alerts_reduced,
        'alert_reduction_pct': round(reduction_pct, 1),
        'false_positives_saved': fp_saved,
    },
    'threshold_sweep': results
}

with open('output/bayesian/bayesian_report.json', 'w') as f:
    json.dump(bayesian_report, f, indent=2)

# Save optimal threshold for use in backend (Phase 3)
torch.save({
    'optimal_threshold' : best_threshold,
    'risk_threshold'    : RISK_THRESHOLD,
    'n_passes'          : 30
}, 'output/bayesian/bayesian_config.pt')

print(f"\n  Saved:")
print(f"  output/bayesian/bayesian_report.json   — full results")
print(f"  output/bayesian/bayesian_config.pt     — config for backend")
print(f"\n  Phase 1 | Week 4 COMPLETE")
print(f"  Phase 1 is now FULLY COMPLETE")
print(f"  NEXT: Phase 2 | Week 5 — XAI explanation layer")
print("=" * 65)