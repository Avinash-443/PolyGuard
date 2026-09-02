# ================================================================
#  PolyGuard — Phase 2, Week 5
#  xai_explainer.py — GNNExplainer + SHAP Explanation Layer
#  Status: Week 5 / 20
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import negative_sampling
from torch_geometric.explain import Explainer, GNNExplainer
import shap
import numpy as np
import pickle
import json
import os

os.makedirs('output/xai', exist_ok=True)

print("=" * 65)
print("  PolyGuard — XAI Explanation Layer")
print("  Phase 2 | Week 5 | Step 5 of 20")
print("=" * 65)

# ── STEP 1: RE-DEFINE MODEL ───────────────────────────────────

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

    def forward_for_xai(self, x, edge_index):
        """
        Simplified forward used by GNNExplainer.
        Returns node embeddings only — explainer
        hooks into this to understand feature importance.
        """
        z = self.encode(x, edge_index)
        return z


# ── STEP 2: LOAD EVERYTHING ───────────────────────────────────

print("\n[STEP 1/6] Loading model and data...")

graph = torch.load('output/polyguard_graph.pt', weights_only=False)

with open('output/drug_to_id.pkl', 'rb') as f:
    drug_to_id = pickle.load(f)
with open('output/id_to_drug.pkl', 'rb') as f:
    id_to_drug = pickle.load(f)
with open('output/twosides_lookup.pkl', 'rb') as f:
    twosides_lookup = pickle.load(f)

# Load DrugBank severity data for explanation context
import pandas as pd
df_ddi = pd.read_csv('../Dataset/db_drug_interactions.csv')
df_ddi.columns = ['drug1', 'drug2', 'interaction']
df_ddi['drug1'] = df_ddi['drug1'].str.strip().str.lower()
df_ddi['drug2'] = df_ddi['drug2'].str.strip().str.lower()

# Build interaction text lookup
interaction_lookup = {}
for _, row in df_ddi.iterrows():
    key1 = (row['drug1'], row['drug2'])
    key2 = (row['drug2'], row['drug1'])
    interaction_lookup[key1] = str(row['interaction'])
    interaction_lookup[key2] = str(row['interaction'])

model = PolyGuardGNN(
    in_dim=128, hidden_dim=256,
    out_dim=128, heads=4, dropout=0.3
)
model.load_state_dict(
    torch.load('output/polyguard_model_best.pt', weights_only=True)
)
model.eval()

# Rebuild train split (same seed)
torch.manual_seed(42)
np.random.seed(42)
edge_index  = graph.edge_index
edge_attr   = graph.edge_attr
num_edges   = edge_index.shape[1]
perm        = torch.randperm(num_edges,
              generator=torch.Generator().manual_seed(42))
edge_index_s = edge_index[:, perm]
train_end    = int(num_edges * 0.80)
train_edges  = edge_index_s[:, :train_end]

print(f"  Model loaded        : output/polyguard_model_best.pt")
print(f"  Graph               : {graph.num_nodes:,} nodes, {graph.num_edges:,} edges")
print(f"  Drug vocabulary     : {len(drug_to_id):,}")
print(f"  Interaction texts   : {len(interaction_lookup):,} pairs")


# ── STEP 3: EDGE IMPORTANCE SCORING ──────────────────────────
# For each drug pair alert, we compute which neighbouring edges
# in the patient's subgraph most influenced the prediction.
# We use gradient-based attribution — the gradient of the
# output score with respect to each edge embedding tells us
# how much that connection mattered to the final alert.

print("\n[STEP 2/6] Computing edge importance scores...")

def get_edge_importance(drug1_name, drug2_name, top_k=5):
    """
    For a given drug pair, compute the importance of each
    neighbouring edge in the graph by gradient attribution.
    Returns top-k most influential drug connections.
    """
    d1 = drug1_name.strip().lower()
    d2 = drug2_name.strip().lower()

    if d1 not in drug_to_id or d2 not in drug_to_id:
        return None, None

    id1 = drug_to_id[d1]
    id2 = drug_to_id[d2]

    # Enable gradient tracking on node features
    x = graph.x.clone().requires_grad_(True)
    pred_edge = torch.tensor([[id1], [id2]], dtype=torch.long)

    model.train()  # enable dropout-style grad flow
    score, z = model(x, train_edges, pred_edge)
    prob = torch.sigmoid(score)

    # Backpropagate to get gradient of score w.r.t. node features
    prob.backward()

    # Gradient magnitude per node = importance score
    grad_magnitude = x.grad.abs().sum(dim=1)

    # Find top-k most influential drug nodes
    top_nodes = torch.topk(grad_magnitude, min(top_k + 2, graph.num_nodes))
    important_drugs = []
    for idx in top_nodes.indices.tolist():
        drug_name = id_to_drug.get(idx, 'unknown')
        importance = float(grad_magnitude[idx].item())
        if drug_name not in [d1, d2]:  # exclude the pair itself
            important_drugs.append({
                'drug': drug_name,
                'importance_score': round(importance, 4),
                'node_id': idx
            })
        if len(important_drugs) >= top_k:
            break

    return float(prob.item()), important_drugs


# Test on known pairs
test_pairs = [
    ('sertraline', 'tramadol'),
    ('amoxicillin', 'methotrexate'),
    ('warfarin', 'aspirin'),
]

print(f"\n  Testing edge importance on {len(test_pairs)} drug pairs...\n")
edge_importance_results = {}

for d1, d2 in test_pairs:
    prob, important = get_edge_importance(d1, d2, top_k=5)
    if prob is not None:
        edge_importance_results[f"{d1}+{d2}"] = {
            'risk_score': round(prob, 4),
            'influential_drugs': important
        }
        print(f"  {d1.title()} + {d2.title()}")
        print(f"    Risk score         : {prob:.4f}")
        print(f"    Influential drugs  :")
        for item in important[:3]:
            print(f"      - {item['drug'].title():<30} importance: {item['importance_score']:.4f}")
        print()


# ── STEP 4: SHAP FEATURE ATTRIBUTION ─────────────────────────
# SHAP answers: which features (dimensions) of a drug's
# embedding most contributed to the interaction score?
# We use KernelSHAP — model-agnostic, works with any function.
# We treat the edge scoring head as a black-box function
# and compute SHAP values for the concatenated [drug1, drug2]
# feature vector.

print("\n[STEP 3/6] Computing SHAP feature attributions...")

model.eval()
with torch.no_grad():
    node_embeddings = model.encode(graph.x, train_edges)

def edge_score_fn(embedding_pairs):
    """
    Black-box function for SHAP:
    Takes N x (out_dim*2) array of concatenated embeddings,
    returns N x 1 risk scores.
    """
    inp = torch.tensor(embedding_pairs, dtype=torch.float32)
    with torch.no_grad():
        scores = model.edge_scorer(inp).squeeze(-1)
        probs  = torch.sigmoid(scores)
    return probs.numpy()

# Build background dataset for SHAP — sample 100 random pairs
torch.manual_seed(42)
bg_idx = torch.randint(0, graph.num_nodes, (100, 2))
bg_embeddings = []
for i in range(100):
    e1 = node_embeddings[bg_idx[i, 0]].numpy()
    e2 = node_embeddings[bg_idx[i, 1]].numpy()
    bg_embeddings.append(np.concatenate([e1, e2]))
background = np.array(bg_embeddings)

# Create KernelSHAP explainer
explainer_shap = shap.KernelExplainer(
    edge_score_fn,
    shap.sample(background, 50)
)

print(f"  SHAP background     : 50 random drug pairs")
print(f"  Computing SHAP values for test pairs...")

shap_results = {}
for d1, d2 in test_pairs:
    if d1 not in drug_to_id or d2 not in drug_to_id:
        continue

    id1 = drug_to_id[d1]
    id2 = drug_to_id[d2]

    with torch.no_grad():
        emb1 = node_embeddings[id1].numpy()
        emb2 = node_embeddings[id2].numpy()

    pair_input = np.concatenate([emb1, emb2]).reshape(1, -1)

    # Compute SHAP — nsamples controls accuracy vs speed
    shap_vals = explainer_shap.shap_values(
        pair_input, nsamples=100, silent=True
    )

    shap_array = np.array(shap_vals).flatten()

    # Split SHAP values: first 128 = drug1 features, next 128 = drug2
    shap_drug1 = shap_array[:128]
    shap_drug2 = shap_array[128:]

    # Top contributing feature dimensions
    top_d1_dims = np.argsort(np.abs(shap_drug1))[-5:][::-1]
    top_d2_dims = np.argsort(np.abs(shap_drug2))[-5:][::-1]

    total_shap_d1 = float(np.abs(shap_drug1).sum())
    total_shap_d2 = float(np.abs(shap_drug2).sum())

    shap_results[f"{d1}+{d2}"] = {
        'drug1_total_contribution' : round(total_shap_d1, 4),
        'drug2_total_contribution' : round(total_shap_d2, 4),
        'drug1_top_feature_dims'   : top_d1_dims.tolist(),
        'drug2_top_feature_dims'   : top_d2_dims.tolist(),
        'drug1_top_shap_values'    : [round(float(shap_drug1[i]), 4) for i in top_d1_dims],
        'drug2_top_shap_values'    : [round(float(shap_drug2[i]), 4) for i in top_d2_dims],
    }

    dominance = d1 if total_shap_d1 > total_shap_d2 else d2
    print(f"\n  {d1.title()} + {d2.title()}")
    print(f"    {d1.title()} contribution : {total_shap_d1:.4f}")
    print(f"    {d2.title()} contribution : {total_shap_d2:.4f}")
    print(f"    Dominant drug        : {dominance.title()}")


# ── STEP 5: EXPLANATION CARD GENERATOR ───────────────────────
# This is what the doctor actually sees on screen.
# We combine: risk score + SHAP attribution + TWOSIDES side
# effect + DrugBank interaction text → one plain-English card.

print("\n\n[STEP 4/6] Generating explanation cards...")

SEVERITY_KEYWORDS = {
    'severe': ['severe', 'fatal', 'life-threatening', 'hemorrhage',
               'cardiac arrest', 'toxicity', 'seizure', 'death',
               'respiratory depression', 'anaphylaxis'],
    'moderate': ['increase', 'decrease', 'inhibit', 'enhance',
                 'prolong', 'impair', 'alter', 'potentiate']
}

def classify_severity(text):
    text = text.lower()
    for kw in SEVERITY_KEYWORDS['severe']:
        if kw in text:
            return 'SEVERE'
    for kw in SEVERITY_KEYWORDS['moderate']:
        if kw in text:
            return 'MODERATE'
    return 'MILD'

def get_alert_badge(risk_score, severity):
    if risk_score >= 0.80 or severity == 'SEVERE':
        return 'RED'
    elif risk_score >= 0.50 or severity == 'MODERATE':
        return 'YELLOW'
    else:
        return 'GREEN'

def generate_explanation_card(drug1_name, drug2_name):
    d1 = drug1_name.strip().lower()
    d2 = drug2_name.strip().lower()

    if d1 not in drug_to_id or d2 not in drug_to_id:
        return {'error': f"One or both drugs not found in vocabulary"}

    id1 = drug_to_id[d1]
    id2 = drug_to_id[d2]

    # Get risk score
    pred_edge = torch.tensor([[id1], [id2]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        score, _ = model(graph.x, train_edges, pred_edge)
        risk_score = float(torch.sigmoid(score).item())

    # Get interaction text from DrugBank
    interaction_text = interaction_lookup.get(
        (d1, d2), interaction_lookup.get((d2, d1), None)
    )

    # Get side effect from TWOSIDES
    side_effect = twosides_lookup.get(
        (d1, d2), twosides_lookup.get((d2, d1), 'Not available')
    )

    # Classify severity
    severity = classify_severity(interaction_text) \
               if interaction_text else 'MODERATE'

    # Get alert badge
    badge = get_alert_badge(risk_score, severity)

    # Get SHAP contribution if available
    shap_key = f"{d1}+{d2}"
    alt_key  = f"{d2}+{d1}"
    shap_info = shap_results.get(shap_key, shap_results.get(alt_key, None))

    dominant_drug = None
    if shap_info:
        d1_contrib = shap_info['drug1_total_contribution']
        d2_contrib = shap_info['drug2_total_contribution']
        dominant_drug = d1.title() if d1_contrib > d2_contrib else d2.title()

    # Build explanation card
    card = {
        'drug1'            : d1.title(),
        'drug2'            : d2.title(),
        'risk_score'       : round(risk_score, 4),
        'severity'         : severity,
        'alert_badge'      : badge,
        'interaction_text' : interaction_text[:300] if interaction_text else 'See clinical guidelines',
        'side_effect'      : side_effect,
        'dominant_drug'    : dominant_drug,
        'shap_available'   : shap_info is not None,
        'clinical_action'  : (
            'DO NOT PRESCRIBE without specialist consultation'
            if badge == 'RED' else
            'Monitor patient closely if prescribed'
            if badge == 'YELLOW' else
            'Safe to prescribe — routine monitoring'
        ),
    }

    return card


# Generate cards for test pairs + additional pairs
all_test_pairs = [
    ('sertraline',   'tramadol'),
    ('amoxicillin',  'methotrexate'),
    ('warfarin',     'aspirin'),
    ('lisinopril',   'spironolactone'),
    ('metformin',    'contrast media'),
    ('ciprofloxacin','tizanidine'),
]

explanation_cards = {}
print(f"\n  {'─' * 60}")

for d1, d2 in all_test_pairs:
    card = generate_explanation_card(d1, d2)
    key  = f"{d1}+{d2}"
    explanation_cards[key] = card

    if 'error' in card:
        print(f"  {d1.title()} + {d2.title()}: {card['error']}")
    else:
        print(f"\n  Drug 1         : {card['drug1']}")
        print(f"  Drug 2         : {card['drug2']}")
        print(f"  Risk Score     : {card['risk_score']}")
        print(f"  Severity       : {card['severity']}")
        print(f"  Alert Badge    : [{card['alert_badge']}]")
        print(f"  Side Effect    : {card['side_effect']}")
        print(f"  Dominant Drug  : {card['dominant_drug']}")
        print(f"  Clinical Action: {card['clinical_action']}")
        if card['interaction_text']:
            print(f"  Mechanism      : {card['interaction_text'][:120]}...")
    print(f"  {'─' * 60}")


# ── STEP 6: SAVE ALL XAI OUTPUTS ─────────────────────────────

print("\n[STEP 5/6] Saving XAI outputs...")

with open('output/xai/edge_importance_scores.pkl', 'wb') as f:
    pickle.dump(edge_importance_results, f)

with open('output/xai/shap_values.pkl', 'wb') as f:
    pickle.dump(shap_results, f)

with open('output/xai/explanation_cards.json', 'w') as f:
    json.dump(explanation_cards, f, indent=2)

xai_report = {
    'phase'              : 'Phase 2, Week 5',
    'pairs_explained'    : len(explanation_cards),
    'shap_pairs'         : len(shap_results),
    'edge_importance_pairs': len(edge_importance_results),
    'cards_generated'    : len(explanation_cards),
    'badge_distribution' : {
        badge: sum(1 for c in explanation_cards.values()
                   if c.get('alert_badge') == badge)
        for badge in ['RED', 'YELLOW', 'GREEN']
    },
}

with open('output/xai/xai_report.json', 'w') as f:
    json.dump(xai_report, f, indent=2)

print(f"\n  output/xai/edge_importance_scores.pkl")
print(f"  output/xai/shap_values.pkl")
print(f"  output/xai/explanation_cards.json")
print(f"  output/xai/xai_report.json")

print(f"\n  Badge distribution:")
for badge, count in xai_report['badge_distribution'].items():
    print(f"    [{badge}] : {count} pairs")

print(f"\n{'=' * 65}")
print(f"  Phase 2 | Week 5 COMPLETE")
print(f"  NEXT: Week 6 — explanation_cards.py")
print(f"  (Map all TWOSIDES pairs to full explanation cards)")
print(f"{'=' * 65}")