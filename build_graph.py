# ================================================================
#  PolyGuard — Phase 1, Week 1
#  build_graph.py — Drug Knowledge Graph Construction
#  Status: Week 1 / 20
# ================================================================

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
import os
import time

start_time = time.time()

print("=" * 65)
print("  PolyGuard — Drug Knowledge Graph Builder")
print("  Phase 1 | Week 1 | Step 1 of 20")
print("=" * 65)

# ── STEP 1: LOAD DATASETS ─────────────────────────────────────

print("\n[STEP 1/5] Loading datasets...")

df_ddi = pd.read_csv('../Dataset/db_drug_interactions.csv')

df_two = pd.read_csv(
    '../Dataset/TWOSIDES.csv.gz',
    compression='gzip',
    low_memory=False
)

df_off = pd.read_csv(
    '../Dataset/OFFSIDES.csv.gz',
    compression='gzip',
    low_memory=False
)

print(f"  DrugBank DDI loaded   : {len(df_ddi):>10,} rows")
print(f"  TWOSIDES loaded       : {len(df_two):>10,} rows")
print(f"  OFFSIDES loaded       : {len(df_off):>10,} rows")

# ── STEP 2: CLEAN DRUGBANK ────────────────────────────────────

print("\n[STEP 2/5] Cleaning and processing DrugBank DDI...")

df_ddi.columns = ['drug1', 'drug2', 'interaction']
df_ddi = df_ddi.dropna()
df_ddi['drug1'] = df_ddi['drug1'].str.strip().str.lower()
df_ddi['drug2'] = df_ddi['drug2'].str.strip().str.lower()
df_ddi = df_ddi.drop_duplicates(subset=['drug1', 'drug2'])

SEVERE_KEYWORDS = [
    'severe', 'serious', 'fatal', 'life-threatening',
    'hemorrhage', 'bleeding', 'cardiac arrest', 'toxicity',
    'arrhythmia', 'seizure', 'renal failure', 'hepatotoxicity',
    'anaphylaxis', 'death', 'coma', 'respiratory depression'
]

MODERATE_KEYWORDS = [
    'increase', 'decrease', 'reduce', 'enhance', 'inhibit',
    'prolong', 'elevate', 'lower', 'impair', 'affect',
    'may', 'can', 'could', 'risk', 'alter', 'potentiate'
]

def assign_severity(text):
    text = str(text).lower()
    if any(kw in text for kw in SEVERE_KEYWORDS):
        return 2  # severe
    elif any(kw in text for kw in MODERATE_KEYWORDS):
        return 1  # moderate
    else:
        return 0  # mild

df_ddi['severity']       = df_ddi['interaction'].apply(assign_severity)
df_ddi['severity_label'] = df_ddi['severity'].map(
    {0: 'mild', 1: 'moderate', 2: 'severe'}
)

severity_counts = df_ddi['severity_label'].value_counts()
print(f"  Clean drug pairs      : {len(df_ddi):>10,}")
print(f"  Severe interactions   : {severity_counts.get('severe',   0):>10,}")
print(f"  Moderate interactions : {severity_counts.get('moderate', 0):>10,}")
print(f"  Mild interactions     : {severity_counts.get('mild',     0):>10,}")

# ── STEP 3: PROCESS TWOSIDES ──────────────────────────────────

print("\n[STEP 3/5] Processing TWOSIDES (high-confidence filter)...")

df_two.columns = df_two.columns.str.strip()
df_two = df_two.dropna(subset=[
    'drug_1_concept_name',
    'drug_2_concept_name',
    'condition_concept_name',
    'PRR'
])

df_two['PRR'] = pd.to_numeric(df_two['PRR'], errors='coerce')
df_two         = df_two.dropna(subset=['PRR'])

# PRR > 2 means this drug combination causes the side effect
# at 2x higher rate than either drug alone — clinically significant
df_two_hc          = df_two[df_two['PRR'] > 2.0].copy()
df_two_hc['drug1'] = df_two_hc['drug_1_concept_name'].str.strip().str.lower()
df_two_hc['drug2'] = df_two_hc['drug_2_concept_name'].str.strip().str.lower()
df_two_hc['side_effect'] = df_two_hc['condition_concept_name'].str.strip()

# Keep top side effect per drug pair (highest PRR = most significant)
df_two_top = (
    df_two_hc
    .sort_values('PRR', ascending=False)
    .drop_duplicates(subset=['drug1', 'drug2'])
    [['drug1', 'drug2', 'side_effect', 'PRR']]
    .reset_index(drop=True)
)

print(f"  Rows after PRR filter : {len(df_two_hc):>10,}")
print(f"  Unique drug pairs     : {len(df_two_top):>10,}")
print(f"  Sample side effect    : {df_two_top['side_effect'].iloc[0]}")

# ── STEP 4: BUILD DRUG VOCAB + EDGES ─────────────────────────

print("\n[STEP 4/5] Building drug vocabulary and graph edges...")

drugs_ddi = set(df_ddi['drug1'].tolist()     + df_ddi['drug2'].tolist())
drugs_two = set(df_two_top['drug1'].tolist() + df_two_top['drug2'].tolist())
all_drugs  = sorted(drugs_ddi.union(drugs_two))

drug_to_id = {drug: idx for idx, drug in enumerate(all_drugs)}
id_to_drug = {idx: drug for drug, idx in drug_to_id.items()}

print(f"  Total unique drugs    : {len(drug_to_id):>10,}")

# Build edge list from DrugBank
edge_src      = []
edge_dst      = []
edge_severity = []
edge_labels   = []

skipped = 0
for _, row in df_ddi.iterrows():
    d1 = row['drug1']
    d2 = row['drug2']
    if d1 in drug_to_id and d2 in drug_to_id:
        s  = drug_to_id[d1]
        t  = drug_to_id[d2]
        sv = int(row['severity'])
        lb = str(row['interaction'])[:150]

        # Forward edge
        edge_src.append(s)
        edge_dst.append(t)
        edge_severity.append(sv)
        edge_labels.append(lb)

        # Reverse edge (undirected graph)
        edge_src.append(t)
        edge_dst.append(s)
        edge_severity.append(sv)
        edge_labels.append(lb)
    else:
        skipped += 1

severe_count   = sum(1 for s in edge_severity if s == 2)
moderate_count = sum(1 for s in edge_severity if s == 1)
mild_count     = sum(1 for s in edge_severity if s == 0)

print(f"  Total edges built     : {len(edge_src):>10,}")
print(f"  Severe edges          : {severe_count:>10,}")
print(f"  Moderate edges        : {moderate_count:>10,}")
print(f"  Mild edges            : {mild_count:>10,}")
print(f"  Skipped pairs         : {skipped:>10,}")

# Build TWOSIDES side-effect lookup table
# Used later for explanation card generation (L4 — XAI layer)
twosides_lookup = {}
for _, row in df_two_top.iterrows():
    key1 = (row['drug1'], row['drug2'])
    key2 = (row['drug2'], row['drug1'])
    twosides_lookup[key1] = row['side_effect']
    twosides_lookup[key2] = row['side_effect']

print(f"  TWOSIDES pairs indexed: {len(twosides_lookup):>10,}")

# ── STEP 5: CREATE PYTORCH GEOMETRIC GRAPH ───────────────────

print("\n[STEP 5/5] Creating PyTorch Geometric graph object...")

num_nodes = len(drug_to_id)

# Node features — 128-dim random initialisation with fixed seed
# These will be replaced by learned embeddings during GNN training
torch.manual_seed(42)
x = torch.randn(num_nodes, 128)

edge_index = torch.tensor(
    [edge_src, edge_dst],
    dtype=torch.long
)

edge_attr = torch.tensor(edge_severity, dtype=torch.long)

graph = Data(
    x          = x,
    edge_index = edge_index,
    edge_attr  = edge_attr,
    num_nodes  = num_nodes
)

print(f"  Graph nodes           : {graph.num_nodes:>10,}")
print(f"  Graph edges           : {graph.num_edges:>10,}")
print(f"  Node feature dim      : {graph.x.shape[1]:>10}")
print(f"  Edge attr shape       : {str(graph.edge_attr.shape):>10}")

# ── SAVE ALL OUTPUTS ──────────────────────────────────────────

print("\n  Saving all outputs to output/ folder...")

os.makedirs('output', exist_ok=True)

# PyG graph object — used directly in train_gnn.py next week
torch.save(graph, 'output/polyguard_graph.pt')

# Drug vocabulary — used in backend for drug name → node ID lookup
with open('output/drug_to_id.pkl', 'wb') as f:
    pickle.dump(drug_to_id, f)

with open('output/id_to_drug.pkl', 'wb') as f:
    pickle.dump(id_to_drug, f)

# TWOSIDES lookup — used in XAI explanation card generation (Month 2)
with open('output/twosides_lookup.pkl', 'wb') as f:
    pickle.dump(twosides_lookup, f)

# CSV files — used for analysis and report tables
df_ddi[['drug1', 'drug2', 'severity',
        'severity_label', 'interaction']].to_csv(
    'output/ddi_with_severity.csv', index=False
)

df_two_top.to_csv(
    'output/twosides_filtered.csv', index=False
)

# ── FINAL SUMMARY ─────────────────────────────────────────────

elapsed = round(time.time() - start_time, 1)

print("\n" + "=" * 65)
print("  SAVED FILES")
print("=" * 65)
print("  output/polyguard_graph.pt       — PyG graph object")
print("  output/drug_to_id.pkl           — drug name to node ID map")
print("  output/id_to_drug.pkl           — node ID to drug name map")
print("  output/twosides_lookup.pkl      — drug pair to side effect")
print("  output/ddi_with_severity.csv    — DrugBank with severity labels")
print("  output/twosides_filtered.csv    — TWOSIDES high-confidence pairs")
print("=" * 65)
print(f"\n  Phase 1 | Week 1 COMPLETE — time taken: {elapsed}s")
print("\n  PROGRESS : Week 1 done  |  Week 2 next — train_gnn.py")
print("  NEXT STEP: Tell Claude you are ready for Week 2")
print("=" * 65)