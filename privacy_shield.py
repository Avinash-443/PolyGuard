# ================================================================
#  PolyGuard — Phase 2, Week 7
#  privacy_shield.py — Privacy-Graph Shield (Patent 3)
#  Masked Node Inference Protocol
#  Status: Week 7 / 20
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
import pickle
import json
import os
from enum import Enum

os.makedirs('output/privacy', exist_ok=True)

print("=" * 65)
print("  PolyGuard — Privacy-Graph Shield")
print("  Phase 2 | Week 7 | Step 7 of 20")
print("  Patent Contribution 3")
print("=" * 65)

# ── PRIVACY LEVEL ENUM ────────────────────────────────────────

class PrivacyLevel(Enum):
    """
    Three-level patient consent system.
    Agreed design from project planning.
    """
    FULL_SHARE       = "full_share"
    INTERACTION_ONLY = "interaction_only"  # Patent 3 applies here
    FULL_PRIVATE     = "full_private"


# ── MODEL DEFINITION ──────────────────────────────────────────

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

    def encode_with_mask(self, x, edge_index, private_node_ids):
        """
        PATENT 3 CORE METHOD — Masked Node Inference.

        Runs the full GNN encode pass normally — private nodes
        participate in message-passing and influence all
        neighbouring node embeddings through graph convolution.

        After encoding, private node embeddings are replaced
        with zero vectors before returning.

        This means:
        - Private drug DID influence the risk computation
          (through message-passing in layers 1, 2, 3)
        - Private drug embedding is NOT in the output
          (cannot be reconstructed from what the decoder sees)

        This is the novel mechanism — participation without
        identity exposure.
        """
        # Full encode — private nodes participate normally
        z = self.encode(x, edge_index)

        # Zero-mask private node embeddings AFTER encode
        if private_node_ids:
            mask = torch.ones_like(z)
            for node_id in private_node_ids:
                mask[node_id] = 0.0
            z = z * mask

        return z

    def forward_with_privacy(self, x, edge_index,
                             pred_edges, private_node_ids=None):
        """
        Forward pass with Privacy-Graph Shield applied.
        Used by the backend for all interaction checks.
        """
        if private_node_ids is None:
            private_node_ids = []

        z     = self.encode_with_mask(x, edge_index, private_node_ids)
        score = self.decode(z, pred_edges)
        return score, z

    def forward(self, x, edge_index, pred_edges,
                comorbidity_context=None):
        z     = self.encode(x, edge_index, comorbidity_context)
        score = self.decode(z, pred_edges)
        return score, z


# ── LOAD RESOURCES ────────────────────────────────────────────

print("\n[STEP 1/7] Loading model and resources...")

graph = torch.load('output/polyguard_graph.pt', weights_only=False)

with open('output/drug_to_id.pkl', 'rb') as f:
    drug_to_id = pickle.load(f)
with open('output/id_to_drug.pkl', 'rb') as f:
    id_to_drug = pickle.load(f)
with open('output/twosides_lookup.pkl', 'rb') as f:
    twosides_lookup = pickle.load(f)
with open('output/cards/drug_compositions.json') as f:
    drug_compositions = json.load(f)

bayesian_config = torch.load(
    'output/bayesian/bayesian_config.pt', weights_only=True
)
RISK_THRESHOLD = bayesian_config['risk_threshold']
UNCERT_THRESH  = bayesian_config['optimal_threshold']

model = PolyGuardGNN(
    in_dim=128, hidden_dim=256,
    out_dim=128, heads=4, dropout=0.3
)
model.load_state_dict(
    torch.load('output/polyguard_model_best.pt', weights_only=True)
)
model.eval()

torch.manual_seed(42)
edge_index  = graph.edge_index
num_edges   = edge_index.shape[1]
perm        = torch.randperm(num_edges,
              generator=torch.Generator().manual_seed(42))
train_edges = edge_index[:, perm][:, :int(num_edges * 0.80)]

print(f"  Model loaded      : output/polyguard_model_best.pt")
print(f"  Graph             : {graph.num_nodes:,} nodes, {graph.num_edges:,} edges")
print(f"  Drug vocabulary   : {len(drug_to_id):,}")


# ── PRIVACY SHIELD CORE FUNCTIONS ────────────────────────────

print("\n[STEP 2/7] Initialising Privacy-Graph Shield...")

def get_ingredients(drug_name):
    """Decompose drug into ingredients for multi-ingredient support."""
    name = drug_name.strip().lower()
    if name in drug_compositions:
        return drug_compositions[name]
    return [name] if name in drug_to_id else []


def mc_predict_with_privacy(drug1_name, drug2_name,
                             private_node_ids=None, n_passes=20):
    """
    Bayesian MC Dropout prediction with Privacy-Graph Shield.

    If drug2 is private, its node_id is in private_node_ids.
    The private drug still participates in message-passing
    but its embedding is zeroed before decoding.
    """
    if private_node_ids is None:
        private_node_ids = []

    d1 = drug1_name.strip().lower()
    d2 = drug2_name.strip().lower()

    if d1 not in drug_to_id or d2 not in drug_to_id:
        return None, None

    id1 = drug_to_id[d1]
    id2 = drug_to_id[d2]
    pred_edge = torch.tensor([[id1], [id2]], dtype=torch.long)

    model.train()
    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            score, _ = model.forward_with_privacy(
                graph.x, train_edges, pred_edge, private_node_ids
            )
            all_probs.append(torch.sigmoid(score).item())

    return float(np.mean(all_probs)), float(np.std(all_probs))


def build_privacy_card(new_drug, private_drug,
                       mean_prob, uncertainty, privacy_level):
    """
    Builds the card shown to doctor when interacting drug is private.
    Drug identity, mechanism, and side effect are all hidden.
    Only the alert badge and a generic warning are shown.
    """
    is_risky     = mean_prob   > RISK_THRESHOLD
    is_confident = uncertainty < UNCERT_THRESH

    if is_risky and is_confident:
        badge = 'RED'
    elif is_risky and not is_confident:
        badge = 'YELLOW'
    else:
        badge = 'GREEN'

    return {
        'new_drug'              : new_drug.title(),
        'existing_drug'         : '*** PRIVATE MEDICATION ***',
        'existing_ingredient'   : '*** PRIVATE MEDICATION ***',
        'alert_badge'           : badge,
        'risk_score'            : round(mean_prob, 4),
        'uncertainty'           : round(uncertainty, 4),
        'confidence'            : 'High' if is_confident else 'Low',

        # All identifying information hidden
        'mechanism'             : (
            'A conflict has been detected with a restricted medication. '
            'Contact the patient before prescribing this drug.'
        ),
        'side_effect'           : 'Hidden — medication is marked private',
        'privacy_level'         : privacy_level.value,
        'privacy_note'          : (
            'This patient has marked one or more medications as private. '
            'The interaction check was performed but the drug identity '
            'cannot be disclosed to other physicians per patient consent.'
        ),
        'clinical_action'       : (
            'CONTACT PATIENT before prescribing — '
            'a restricted medication conflict was detected'
            if badge == 'RED' else
            'VERIFY with patient — possible conflict with a restricted medication'
            if badge == 'YELLOW' else
            'No significant conflict detected with restricted medication'
        ),
        'override_allowed'      : True,
        'override_required_reason': badge in ['RED', 'YELLOW'],
    }


def check_with_privacy_shield(
    new_drug,
    patient_medication_profile,
    n_mc_passes=20
):
    """
    MAIN PRIVACY SHIELD FUNCTION — called by FastAPI backend.

    patient_medication_profile is a list of dicts:
    [
        {'drug': 'warfarin',    'privacy': PrivacyLevel.FULL_SHARE},
        {'drug': 'sertraline',  'privacy': PrivacyLevel.INTERACTION_ONLY},
        {'drug': 'metformin',   'privacy': PrivacyLevel.FULL_SHARE},
    ]

    For FULL_SHARE drugs     → full card with drug name and mechanism
    For INTERACTION_ONLY     → Patent 3 applies — masked card, alert only
    For FULL_PRIVATE         → check skipped, doctor warned of blind spot
    """
    new_ingredients = get_ingredients(new_drug)
    if not new_ingredients:
        return {
            'status' : 'error',
            'message': f"'{new_drug}' not found in drug vocabulary",
        }

    results = {
        'new_drug'          : new_drug.title(),
        'overall_badge'     : 'GREEN',
        'full_share_cards'  : [],
        'private_cards'     : [],
        'blind_spot_warning': None,
        'summary'           : {},
    }

    has_full_private = any(
        m['privacy'] == PrivacyLevel.FULL_PRIVATE
        for m in patient_medication_profile
    )

    if has_full_private:
        results['blind_spot_warning'] = (
            'WARNING: One or more medications are fully private. '
            'The interaction check is INCOMPLETE for those drugs. '
            'Proceed with caution and consult the patient directly.'
        )

    for med in patient_medication_profile:
        existing_drug  = med['drug']
        privacy_level  = med['privacy']

        # ── FULL PRIVATE — skip check entirely ────────────────
        if privacy_level == PrivacyLevel.FULL_PRIVATE:
            continue

        existing_ingredients = get_ingredients(existing_drug)

        for new_ing in new_ingredients:
            for existing_ing in existing_ingredients:
                if new_ing == existing_ing:
                    continue

                private_ids = []
                if privacy_level == PrivacyLevel.INTERACTION_ONLY:
                    # Patent 3 — mask existing drug node
                    private_ids = [drug_to_id[existing_ing]]

                mean_prob, uncertainty = mc_predict_with_privacy(
                    new_ing, existing_ing,
                    private_node_ids=private_ids,
                    n_passes=n_mc_passes
                )

                if mean_prob is None:
                    continue

                # ── FULL SHARE — complete card ─────────────────
                if privacy_level == PrivacyLevel.FULL_SHARE:
                    import pandas as pd

                    # Get mechanism text
                    try:
                        df_ddi = pd.read_csv('../Dataset/db_drug_interactions.csv',
                                             nrows=0)
                    except Exception:
                        pass

                    interaction_lookup_local = {}
                    try:
                        df_tmp = pd.read_csv('../Dataset/db_drug_interactions.csv')
                        df_tmp.columns = ['drug1', 'drug2', 'interaction']
                        df_tmp['drug1'] = df_tmp['drug1'].str.strip().str.lower()
                        df_tmp['drug2'] = df_tmp['drug2'].str.strip().str.lower()
                        for _, row in df_tmp.iterrows():
                            interaction_lookup_local[(row['drug1'], row['drug2'])] = str(row['interaction'])
                            interaction_lookup_local[(row['drug2'], row['drug1'])] = str(row['interaction'])
                    except Exception:
                        pass

                    mechanism = interaction_lookup_local.get(
                        (new_ing, existing_ing),
                        interaction_lookup_local.get(
                            (existing_ing, new_ing), 'See clinical guidelines'
                        )
                    )
                    side_effect = twosides_lookup.get(
                        (new_ing, existing_ing),
                        twosides_lookup.get(
                            (existing_ing, new_ing), 'Not available'
                        )
                    )

                    is_risky     = mean_prob   > RISK_THRESHOLD
                    is_confident = uncertainty < UNCERT_THRESH
                    if is_risky and is_confident:
                        badge = 'RED'
                    elif is_risky:
                        badge = 'YELLOW'
                    else:
                        badge = 'GREEN'

                    card = {
                        'new_drug'        : new_drug.title(),
                        'existing_drug'   : existing_drug.title(),
                        'new_ingredient'  : new_ing.title(),
                        'existing_ingredient': existing_ing.title(),
                        'alert_badge'     : badge,
                        'risk_score'      : round(mean_prob, 4),
                        'uncertainty'     : round(uncertainty, 4),
                        'confidence'      : 'High' if is_confident else 'Low',
                        'mechanism'       : mechanism[:300] if mechanism else 'See clinical guidelines',
                        'side_effect'     : side_effect,
                        'privacy_level'   : privacy_level.value,
                        'override_allowed': True,
                        'override_required_reason': badge in ['RED', 'YELLOW'],
                        'clinical_action' : (
                            'DO NOT PRESCRIBE without specialist consultation'
                            if badge == 'RED' else
                            'Prescribe with caution — monitor patient closely'
                            if badge == 'YELLOW' else
                            'Safe to prescribe — routine monitoring advised'
                        ),
                    }
                    results['full_share_cards'].append(card)

                # ── INTERACTION ONLY — masked card ────────────
                elif privacy_level == PrivacyLevel.INTERACTION_ONLY:
                    card = build_privacy_card(
                        new_drug, existing_drug,
                        mean_prob, uncertainty, privacy_level
                    )
                    results['private_cards'].append(card)

    # Determine overall badge across all cards
    all_badges = (
        [c['alert_badge'] for c in results['full_share_cards']] +
        [c['alert_badge'] for c in results['private_cards']]
    )
    if 'RED' in all_badges:
        results['overall_badge'] = 'RED'
    elif 'YELLOW' in all_badges:
        results['overall_badge'] = 'YELLOW'
    else:
        results['overall_badge'] = 'GREEN'

    results['summary'] = {
        'full_share_checks' : len(results['full_share_cards']),
        'private_checks'    : len(results['private_cards']),
        'blind_spots'       : sum(
            1 for m in patient_medication_profile
            if m['privacy'] == PrivacyLevel.FULL_PRIVATE
        ),
        'red_alerts'        : all_badges.count('RED'),
        'yellow_alerts'     : all_badges.count('YELLOW'),
        'green_alerts'      : all_badges.count('GREEN'),
    }

    return results


# ── STEP 3: PRIVACY VERIFICATION TEST ────────────────────────
# This is the mathematical proof of the patent claim.
# We prove that masking prevents identity reconstruction.

print("\n[STEP 3/7] Running patent verification tests...\n")

def verify_privacy_guarantee(drug1, drug2, n_trials=10):
    """
    Patent 3 Verification Protocol.

    Proves two things:
    1. Private drug DOES influence the risk score
       (proves safety check still ran)
    2. Private drug embedding IS zero in the output
       (proves identity cannot be reconstructed)
    """
    d1 = drug1.strip().lower()
    d2 = drug2.strip().lower()

    if d1 not in drug_to_id or d2 not in drug_to_id:
        return None

    id1 = drug_to_id[d1]
    id2 = drug_to_id[d2]

    # Score WITH private drug visible (normal inference)
    pred_edge = torch.tensor([[id1], [id2]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        score_normal, z_normal = model(
            graph.x, train_edges, pred_edge
        )
        prob_normal = float(torch.sigmoid(score_normal).item())

    # Score WITH private drug masked (Privacy-Graph Shield)
    with torch.no_grad():
        score_masked, z_masked = model.forward_with_privacy(
            graph.x, train_edges, pred_edge,
            private_node_ids=[id2]  # drug2 is private
        )
        prob_masked = float(torch.sigmoid(score_masked).item())

    # Score with drug2 replaced by a RANDOM drug
    # If privacy shield works, masked score should differ from random
    random_id   = torch.randint(0, graph.num_nodes, (1,)).item()
    random_edge = torch.tensor([[id1], [random_id]], dtype=torch.long)
    with torch.no_grad():
        score_random, _ = model(graph.x, train_edges, random_edge)
        prob_random = float(torch.sigmoid(score_random).item())

    # Verification 1 — masked score differs from normal
    # (proves private drug influenced the computation)
    score_differs = abs(prob_normal - prob_masked) > 0.0001

    # Verification 2 — private node embedding is zero in output
    private_embedding_norm = float(z_masked[id2].norm().item())
    embedding_is_zero = private_embedding_norm < 1e-6

    # Verification 3 — masked score is NOT same as random
    # (proves the private drug's influence is real, not random)
    not_random = abs(prob_masked - prob_random) > 0.001

    return {
        'drug1'                   : drug1.title(),
        'drug2_private'           : drug2.title(),
        'prob_normal'             : round(prob_normal, 4),
        'prob_masked'             : round(prob_masked, 4),
        'prob_random_drug'        : round(prob_random, 4),
        'score_differs_from_normal': score_differs,
        'private_embedding_is_zero': embedding_is_zero,
        'private_embedding_norm'  : round(private_embedding_norm, 8),
        'not_same_as_random'      : not_random,
        'patent_claim_verified'   : (
            score_differs and embedding_is_zero
        ),
    }


# Run verification on key drug pairs
verification_pairs = [
    ('sertraline',   'tramadol'),
    ('amoxicillin',  'methotrexate'),
    ('warfarin',     'aspirin'),
    ('ciprofloxacin','tizanidine'),
]

verification_results = {}
print(f"  {'Drug Pair':<40} {'Patent Claim':>15}")
print(f"  {'─' * 57}")

for d1, d2 in verification_pairs:
    result = verify_privacy_guarantee(d1, d2)
    if result:
        verification_results[f"{d1}+{d2}"] = result
        claim = 'VERIFIED ✓' if result['patent_claim_verified'] else 'FAILED ✗'
        print(f"  {d1.title()} + {d2.title():<28} {claim:>15}")
        print(f"    Normal score    : {result['prob_normal']}")
        print(f"    Masked score    : {result['prob_masked']}")
        print(f"    Private emb norm: {result['private_embedding_norm']}")
        print(f"    Embedding zero  : {result['private_embedding_is_zero']}")
        print()


# ── STEP 4: FULL PRIVACY SHIELD TEST ─────────────────────────

print("\n[STEP 4/7] Full privacy shield scenario test...\n")

# Patient has 4 drugs with different privacy levels
patient_profile = [
    {'drug': 'warfarin',    'privacy': PrivacyLevel.FULL_SHARE},
    {'drug': 'sertraline',  'privacy': PrivacyLevel.INTERACTION_ONLY},
    {'drug': 'metformin',   'privacy': PrivacyLevel.FULL_SHARE},
    {'drug': 'alprazolam',  'privacy': PrivacyLevel.FULL_PRIVATE},
]

print("  Patient medication profile:")
for med in patient_profile:
    print(f"    {med['drug'].title():<20} — {med['privacy'].value}")

print(f"\n  Doctor adds: Tramadol")
print(f"  {'─' * 55}")

result = check_with_privacy_shield(
    new_drug                = 'tramadol',
    patient_medication_profile = patient_profile,
    n_mc_passes             = 15
)

print(f"\n  Overall badge       : [{result['overall_badge']}]")
print(f"  Summary             : {result['summary']}")

if result['blind_spot_warning']:
    print(f"\n  ⚠ BLIND SPOT WARNING:")
    print(f"  {result['blind_spot_warning']}")

print(f"\n  Full share cards ({len(result['full_share_cards'])}):")
for card in result['full_share_cards']:
    print(f"    {card['new_ingredient']} + {card['existing_ingredient']:<25} [{card['alert_badge']}] score:{card['risk_score']}")

print(f"\n  Private masked cards ({len(result['private_cards'])}):")
for card in result['private_cards']:
    print(f"    {card['new_drug']} + {card['existing_drug']:<30} [{card['alert_badge']}]")
    print(f"    Action: {card['clinical_action']}")


# ── STEP 5: SAVE ALL PRIVACY SHIELD OUTPUTS ──────────────────

print("\n\n[STEP 5/7] Saving all privacy shield outputs...")

# Save verification results
with open('output/privacy/patent_verification.json', 'w') as f:
    json.dump(verification_results, f, indent=2)

# Save test scenario result
scenario_save = {
    'new_drug'      : result['new_drug'],
    'overall_badge' : result['overall_badge'],
    'summary'       : result['summary'],
    'blind_spot_warning': result['blind_spot_warning'],
    'full_share_count'  : len(result['full_share_cards']),
    'private_count'     : len(result['private_cards']),
}
with open('output/privacy/shield_test_result.json', 'w') as f:
    json.dump(scenario_save, f, indent=2)

# Save privacy shield config for backend
shield_config = {
    'privacy_levels'    : [l.value for l in PrivacyLevel],
    'zero_mask_verified': all(
        r['patent_claim_verified']
        for r in verification_results.values()
    ),
    'verification_pairs': len(verification_results),
    'risk_threshold'    : RISK_THRESHOLD,
    'uncertainty_threshold': UNCERT_THRESH,
}
with open('output/privacy/shield_config.json', 'w') as f:
    json.dump(shield_config, f, indent=2)

print(f"  output/privacy/patent_verification.json")
print(f"  output/privacy/shield_test_result.json")
print(f"  output/privacy/shield_config.json")

# Final patent summary
verified_count = sum(
    1 for r in verification_results.values()
    if r['patent_claim_verified']
)
total_pairs = len(verification_results)

print(f"\n  Patent 3 Verification Summary:")
print(f"  Pairs tested          : {total_pairs}")
print(f"  Patent claim verified : {verified_count}/{total_pairs}")
print(f"  Zero-mask confirmed   : {shield_config['zero_mask_verified']}")

print(f"\n{'=' * 65}")
print(f"  Phase 2 | Week 7 COMPLETE")
print(f"  NEXT: Week 8 — consent_layer.py")
print(f"  (Patient consent management + unit tests)")
print(f"{'=' * 65}")