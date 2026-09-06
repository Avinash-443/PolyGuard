# ================================================================
#  PolyGuard — Phase 2, Week 6
#  explanation_cards.py — Full Explanation Card System
#  Integrated: Multi-ingredient support, Comorbidity context
#  Status: Week 6 / 20
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
import pickle
import json
import os
import pandas as pd
import re

os.makedirs('output/cards', exist_ok=True)

print("=" * 65)
print("  PolyGuard — Explanation Card System")
print("  Phase 2 | Week 6 | Step 6 of 20")
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


# ── STEP 2: LOAD ALL RESOURCES ────────────────────────────────

print("\n[STEP 1/7] Loading all resources...")

graph = torch.load('output/polyguard_graph.pt', weights_only=False)

with open('output/drug_to_id.pkl', 'rb') as f:
    drug_to_id = pickle.load(f)
with open('output/id_to_drug.pkl', 'rb') as f:
    id_to_drug = pickle.load(f)
with open('output/twosides_lookup.pkl', 'rb') as f:
    twosides_lookup = pickle.load(f)
with open('output/bayesian/bayesian_config.pt', 'rb') as f:
    bayesian_config = torch.load(
        'output/bayesian/bayesian_config.pt', weights_only=True
    )

RISK_THRESHOLD  = bayesian_config['risk_threshold']
UNCERT_THRESH   = bayesian_config['optimal_threshold']
N_PASSES        = bayesian_config['n_passes']

df_ddi = pd.read_csv('../Dataset/db_drug_interactions.csv')
df_ddi.columns = ['drug1', 'drug2', 'interaction']
df_ddi['drug1'] = df_ddi['drug1'].str.strip().str.lower()
df_ddi['drug2'] = df_ddi['drug2'].str.strip().str.lower()

interaction_lookup = {}
for _, row in df_ddi.iterrows():
    interaction_lookup[(row['drug1'], row['drug2'])] = str(row['interaction'])
    interaction_lookup[(row['drug2'], row['drug1'])] = str(row['interaction'])

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

print(f"  Graph               : {graph.num_nodes:,} nodes")
print(f"  Drug vocabulary     : {len(drug_to_id):,} drugs")
print(f"  Interaction texts   : {len(interaction_lookup):,} pairs")
print(f"  Bayesian thresholds : risk>{RISK_THRESHOLD}, uncertainty<{UNCERT_THRESH}")


# ── STEP 3: MULTI-INGREDIENT DRUG DICTIONARY ──────────────────
# Built from DrugBank vocabulary — no external dataset needed.
# Covers the most common multi-ingredient drugs in clinical use.
# When a doctor adds a tablet, we decompose it into ingredients
# and check each ingredient against the patient's drug list.

print("\n[STEP 2/7] Loading multi-ingredient drug dictionary...")

MULTI_INGREDIENT_DRUGS = {
    # Antibiotic combinations
    'co-amoxiclav'              : ['amoxicillin', 'clavulanate'],
    'augmentin'                 : ['amoxicillin', 'clavulanate'],
    'trimethoprim-sulfamethoxazole': ['trimethoprim', 'sulfamethoxazole'],
    'co-trimoxazole'            : ['trimethoprim', 'sulfamethoxazole'],
    'piperacillin-tazobactam'   : ['piperacillin', 'tazobactam'],

    # Cardiovascular combinations
    'amlodipine-atorvastatin'   : ['amlodipine', 'atorvastatin'],
    'perindopril-amlodipine'    : ['perindopril', 'amlodipine'],
    'bisoprolol-hydrochlorothiazide': ['bisoprolol', 'hydrochlorothiazide'],
    'valsartan-hydrochlorothiazide' : ['valsartan', 'hydrochlorothiazide'],
    'losartan-hydrochlorothiazide'  : ['losartan', 'hydrochlorothiazide'],
    'lisinopril-hydrochlorothiazide': ['lisinopril', 'hydrochlorothiazide'],

    # Pain and anti-inflammatory
    'co-codamol'                : ['codeine', 'paracetamol'],
    'co-dydramol'               : ['dihydrocodeine', 'paracetamol'],
    'aspirin-dipyridamole'      : ['aspirin', 'dipyridamole'],

    # Diabetes combinations
    'metformin-sitagliptin'     : ['metformin', 'sitagliptin'],
    'metformin-glipizide'       : ['metformin', 'glipizide'],
    'metformin-pioglitazone'    : ['metformin', 'pioglitazone'],

    # Respiratory
    'salbutamol-ipratropium'    : ['salbutamol', 'ipratropium'],
    'fluticasone-salmeterol'    : ['fluticasone', 'salmeterol'],
    'budesonide-formoterol'     : ['budesonide', 'formoterol'],

    # Psychiatric
    'olanzapine-fluoxetine'     : ['olanzapine', 'fluoxetine'],
    'amitriptyline-chlordiazepoxide': ['amitriptyline', 'chlordiazepoxide'],

    # Antihypertensive triple combinations
    'amlodipine-valsartan-hydrochlorothiazide': ['amlodipine', 'valsartan', 'hydrochlorothiazide'],
}

# Filter to only include ingredients present in drug vocabulary
VALIDATED_COMPOSITIONS = {}
for tablet, ingredients in MULTI_INGREDIENT_DRUGS.items():
    valid = [ing for ing in ingredients if ing in drug_to_id]
    if valid:
        VALIDATED_COMPOSITIONS[tablet] = valid

print(f"  Multi-ingredient drugs loaded : {len(VALIDATED_COMPOSITIONS)}")
print(f"  Sample: co-amoxiclav → {VALIDATED_COMPOSITIONS.get('co-amoxiclav', 'not found')}")


# ── STEP 4: COMORBIDITY RISK PROFILES ─────────────────────────
# This is where CC-DGNN comorbidity conditioning becomes visible
# to the doctor. Each comorbidity has associated drug properties
# that increase risk — used to add a warning section to the card.

COMORBIDITY_RISK_PROFILES = {
    'kidney_failure': {
        'display_name'  : 'Chronic Kidney Disease / Renal Failure',
        'risk_keywords' : ['renal', 'kidney', 'nephro', 'creatinine',
                          'excretion', 'clearance', 'glomerular'],
        'warning'       : 'Reduced renal clearance — drug accumulation risk is elevated. Dose reduction may be required.',
        'risk_multiplier': 1.4,
    },
    'liver_disease': {
        'display_name'  : 'Liver Disease / Hepatic Impairment',
        'risk_keywords' : ['hepato', 'liver', 'cyp', 'metabolism',
                          'hepatic', 'enzyme', 'cytochrome'],
        'warning'       : 'Impaired hepatic metabolism — CYP enzyme activity is reduced. Interaction risk is significantly elevated.',
        'risk_multiplier': 1.5,
    },
    'diabetes': {
        'display_name'  : 'Diabetes Mellitus',
        'risk_keywords' : ['glucose', 'glycemic', 'insulin', 'hypoglycemia',
                          'blood sugar', 'diabetes'],
        'warning'       : 'Glycemic control may be affected by this drug combination. Blood sugar monitoring is advised.',
        'risk_multiplier': 1.2,
    },
    'heart_failure': {
        'display_name'  : 'Heart Failure / Cardiac Disease',
        'risk_keywords' : ['cardiac', 'heart', 'qt', 'arrhythmia',
                          'bradycardia', 'tachycardia', 'myocardial'],
        'warning'       : 'Cardiac conduction may be affected. QTc prolongation risk is elevated in heart failure patients.',
        'risk_multiplier': 1.5,
    },
    'hypertension': {
        'display_name'  : 'Hypertension',
        'risk_keywords' : ['blood pressure', 'hypotension', 'antihypertensive',
                          'vasodilation', 'vasoconstriction'],
        'warning'       : 'Blood pressure control may be disrupted by this combination. Monitor BP closely.',
        'risk_multiplier': 1.2,
    },
    'copd': {
        'display_name'  : 'COPD / Respiratory Disease',
        'risk_keywords' : ['broncho', 'respiratory', 'pulmonary',
                          'breathing', 'airway'],
        'warning'       : 'Respiratory function may be further compromised by this drug combination.',
        'risk_multiplier': 1.3,
    },
}


# ── STEP 5: CORE FUNCTIONS ────────────────────────────────────

print("\n[STEP 3/7] Building core card generation functions...")

def get_ingredients(drug_name):
    """
    Decomposes a multi-ingredient drug into its components.
    Returns list of validated ingredient names.
    Falls back to the drug itself if not a known combination.
    """
    name = drug_name.strip().lower()
    if name in VALIDATED_COMPOSITIONS:
        return VALIDATED_COMPOSITIONS[name]
    return [name] if name in drug_to_id else []


def mc_predict(drug1_name, drug2_name, n_passes=20):
    """
    Bayesian MC Dropout prediction for a single drug pair.
    Returns (mean_prob, uncertainty).
    Uses the optimal thresholds from Week 4.
    """
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
            score, _ = model(graph.x, train_edges, pred_edge)
            all_probs.append(torch.sigmoid(score).item())

    mean_prob   = float(np.mean(all_probs))
    uncertainty = float(np.std(all_probs))
    return mean_prob, uncertainty


def classify_severity(text):
    """Classify interaction severity from mechanism text."""
    if not text:
        return 'MODERATE'
    text = text.lower()
    severe_kw = ['severe', 'fatal', 'life-threatening', 'hemorrhage',
                 'cardiac arrest', 'toxicity', 'seizure', 'death',
                 'respiratory depression', 'anaphylaxis', 'coma']
    for kw in severe_kw:
        if kw in text:
            return 'SEVERE'
    return 'MODERATE'


def get_alert_badge(mean_prob, uncertainty, severity):
    """
    Dual-threshold alert gate from Patent 2 (Bayesian layer).
    HIGH confidence + HIGH risk = RED (shown to doctor)
    HIGH risk but LOW confidence = YELLOW (uncertain, verify)
    LOW risk = GREEN
    """
    is_risky     = mean_prob   > RISK_THRESHOLD
    is_confident = uncertainty < UNCERT_THRESH

    if is_risky and is_confident:
        if severity == 'SEVERE':
            return 'RED'
        return 'RED'
    elif is_risky and not is_confident:
        return 'YELLOW'
    else:
        return 'GREEN'


def check_comorbidity_relevance(interaction_text, patient_comorbidities):
    """
    Checks if the interaction mechanism is particularly dangerous
    given the patient's known comorbidities.
    Returns list of comorbidity warning objects.
    CC-DGNN Patent 1 — this is what makes risk patient-specific.
    """
    warnings = []
    if not interaction_text or not patient_comorbidities:
        return warnings

    text_lower = interaction_text.lower()

    for condition in patient_comorbidities:
        condition_key = condition.lower().replace(' ', '_').replace('-', '_')
        profile = COMORBIDITY_RISK_PROFILES.get(condition_key)
        if not profile:
            # Try partial match
            for key, prof in COMORBIDITY_RISK_PROFILES.items():
                if key in condition_key or condition_key in key:
                    profile = prof
                    break

        if profile:
            # Check if interaction text mentions this comorbidity's keywords
            for keyword in profile['risk_keywords']:
                if keyword in text_lower:
                    warnings.append({
                        'condition'     : profile['display_name'],
                        'warning'       : profile['warning'],
                        'risk_multiplier': profile['risk_multiplier'],
                    })
                    break  # one warning per comorbidity

    return warnings


def generate_full_card(
    drug_name,
    patient_drugs,
    patient_comorbidities=None,
    n_mc_passes=20
):
    """
    MAIN CARD GENERATOR — called by FastAPI backend in Week 10.

    Parameters:
    - drug_name            : new drug the doctor is about to prescribe
    - patient_drugs        : list of drug names currently in patient profile
    - patient_comorbidities: list of patient conditions (e.g. ['diabetes', 'kidney_failure'])
    - n_mc_passes          : MC Dropout passes for Bayesian uncertainty

    Returns:
    - A complete card dict for each existing drug the new drug interacts with
    - Includes multi-ingredient decomposition
    - Includes comorbidity-specific warnings
    - Includes override flag for RED alerts
    """
    if patient_comorbidities is None:
        patient_comorbidities = []

    # Decompose new drug into ingredients
    new_ingredients = get_ingredients(drug_name)
    if not new_ingredients:
        return {
            'status' : 'error',
            'message': f"'{drug_name}' not found in drug vocabulary",
            'drug'   : drug_name,
        }

    all_interactions = []

    # Check each new ingredient against each existing patient drug
    for new_ing in new_ingredients:
        for existing_drug in patient_drugs:

            # Decompose existing drug too (may also be multi-ingredient)
            existing_ingredients = get_ingredients(existing_drug)
            if not existing_ingredients:
                continue

            for existing_ing in existing_ingredients:
                if new_ing == existing_ing:
                    continue  # same drug, skip

                # Get Bayesian prediction
                mean_prob, uncertainty = mc_predict(new_ing, existing_ing, n_mc_passes)
                if mean_prob is None:
                    continue

                # Get interaction text
                interaction_text = interaction_lookup.get(
                    (new_ing, existing_ing),
                    interaction_lookup.get((existing_ing, new_ing), None)
                )

                # Get side effect from TWOSIDES
                side_effect = twosides_lookup.get(
                    (new_ing, existing_ing),
                    twosides_lookup.get((existing_ing, new_ing), 'Not in TWOSIDES')
                )

                # Classify severity and get badge
                severity = classify_severity(interaction_text)
                badge    = get_alert_badge(mean_prob, uncertainty, severity)

                # Check comorbidity relevance (CC-DGNN Patent 1)
                comorbidity_warnings = check_comorbidity_relevance(
                    interaction_text, patient_comorbidities
                )

                # Apply comorbidity risk multiplier to score
                adjusted_score = mean_prob
                if comorbidity_warnings:
                    max_multiplier = max(
                        w['risk_multiplier'] for w in comorbidity_warnings
                    )
                    adjusted_score = min(mean_prob * max_multiplier, 1.0)
                    # Re-evaluate badge with adjusted score
                    badge = get_alert_badge(adjusted_score, uncertainty, severity)

                # Build the full card
                card = {
                    # Drug information
                    'new_drug'              : drug_name.title(),
                    'new_ingredient'        : new_ing.title(),
                    'existing_drug'         : existing_drug.title(),
                    'existing_ingredient'   : existing_ing.title(),
                    'is_multi_ingredient'   : len(new_ingredients) > 1 or
                                             len(existing_ingredients) > 1,

                    # Risk assessment
                    'risk_score'            : round(mean_prob, 4),
                    'adjusted_risk_score'   : round(adjusted_score, 4),
                    'uncertainty'           : round(uncertainty, 4),
                    'confidence'            : 'High' if uncertainty < UNCERT_THRESH else 'Low',
                    'severity'              : severity,
                    'alert_badge'           : badge,

                    # Explanation content
                    'mechanism'             : (interaction_text[:300]
                                              if interaction_text
                                              else 'Consult clinical guidelines'),
                    'side_effect'           : side_effect,

                    # Comorbidity context (CC-DGNN Patent 1)
                    'comorbidity_warnings'  : comorbidity_warnings,
                    'comorbidity_adjusted'  : len(comorbidity_warnings) > 0,

                    # Clinical guidance
                    'clinical_action'       : (
                        'DO NOT PRESCRIBE without specialist consultation'
                        if badge == 'RED' else
                        'Prescribe with caution — monitor patient closely'
                        if badge == 'YELLOW' else
                        'Safe to prescribe — routine monitoring advised'
                    ),

                    # Override information (for doctor dashboard)
                    'override_allowed'      : True,
                    'override_required_reason': badge in ['RED', 'YELLOW'],
                    'privacy_note'          : None,  # set by backend if drug is private
                }

                all_interactions.append(card)

    if not all_interactions:
        return {
            'status'        : 'no_interactions',
            'drug'          : drug_name.title(),
            'message'       : 'No known interactions found in vocabulary',
            'alert_badge'   : 'GREEN',
            'override_allowed': True,
        }

    # Sort by adjusted risk score descending
    all_interactions.sort(
        key=lambda x: x['adjusted_risk_score'], reverse=True
    )

    # Summary for the overall prescription check
    highest_badge = 'GREEN'
    for card in all_interactions:
        if card['alert_badge'] == 'RED':
            highest_badge = 'RED'
            break
        elif card['alert_badge'] == 'YELLOW':
            highest_badge = 'YELLOW'

    return {
        'status'            : 'ok',
        'new_drug'          : drug_name.title(),
        'overall_badge'     : highest_badge,
        'total_interactions': len(all_interactions),
        'red_count'         : sum(1 for c in all_interactions if c['alert_badge'] == 'RED'),
        'yellow_count'      : sum(1 for c in all_interactions if c['alert_badge'] == 'YELLOW'),
        'green_count'       : sum(1 for c in all_interactions if c['alert_badge'] == 'GREEN'),
        'interactions'      : all_interactions,
        'override_allowed'  : True,
    }


# ── STEP 6: TEST THE CARD GENERATOR ──────────────────────────

print("\n[STEP 4/7] Testing full card generator...\n")

# ── Test 1: Simple single-ingredient check ────────────────────
print("  TEST 1 — Single drug, no comorbidities")
print("  Adding Sertraline to patient taking [Tramadol, Lisinopril, Metformin]")
print()

result1 = generate_full_card(
    drug_name             = 'sertraline',
    patient_drugs         = ['tramadol', 'lisinopril', 'metformin'],
    patient_comorbidities = [],
    n_mc_passes           = 15
)

print(f"  Overall badge   : [{result1['overall_badge']}]")
print(f"  Total checks    : {result1['total_interactions']}")
print(f"  RED alerts      : {result1['red_count']}")
print(f"  YELLOW alerts   : {result1['yellow_count']}")
if result1['interactions']:
    top = result1['interactions'][0]
    print(f"  Top interaction : {top['new_ingredient']} + {top['existing_ingredient']}")
    print(f"  Risk score      : {top['risk_score']}")
    print(f"  Mechanism       : {top['mechanism'][:100]}...")

# ── Test 2: With comorbidity context ─────────────────────────
print("\n  TEST 2 — Same drug, patient has kidney_failure + heart_failure")
print()

result2 = generate_full_card(
    drug_name             = 'sertraline',
    patient_drugs         = ['tramadol', 'lisinopril', 'metformin'],
    patient_comorbidities = ['kidney_failure', 'heart_failure'],
    n_mc_passes           = 15
)

print(f"  Overall badge   : [{result2['overall_badge']}]")
print(f"  Total checks    : {result2['total_interactions']}")
if result2['interactions']:
    top2 = result2['interactions'][0]
    print(f"  Adjusted score  : {top2['adjusted_risk_score']} "
          f"(was {top2['risk_score']})")
    print(f"  Comorbidity adj : {top2['comorbidity_adjusted']}")
    if top2['comorbidity_warnings']:
        for w in top2['comorbidity_warnings']:
            print(f"  ⚠ {w['condition']}: {w['warning'][:80]}...")

# ── Test 3: Multi-ingredient tablet ──────────────────────────
print("\n  TEST 3 — Multi-ingredient: Co-amoxiclav (Amoxicillin + Clavulanate)")
print("  Patient already taking Methotrexate")
print()

result3 = generate_full_card(
    drug_name             = 'co-amoxiclav',
    patient_drugs         = ['methotrexate', 'warfarin'],
    patient_comorbidities = ['liver_disease'],
    n_mc_passes           = 15
)

print(f"  Overall badge       : [{result3['overall_badge']}]")
print(f"  Total checks        : {result3['total_interactions']}")
print(f"  Multi-ingredient    : {result3['interactions'][0]['is_multi_ingredient'] if result3['interactions'] else 'N/A'}")
if result3['interactions']:
    for card in result3['interactions'][:3]:
        print(f"\n  {card['new_ingredient']} + {card['existing_ingredient']}")
        print(f"    Badge       : [{card['alert_badge']}]  "
              f"Score: {card['adjusted_risk_score']}")
        if card['comorbidity_warnings']:
            print(f"    Comorbidity : {card['comorbidity_warnings'][0]['condition']}")

# ── Test 4: Privacy shield scenario ──────────────────────────
print("\n  TEST 4 — Privacy shield: patient has a PRIVATE drug")
print("  Doctor adds Tramadol, but patient has Sertraline marked PRIVATE")
print()

result4 = generate_full_card(
    drug_name             = 'tramadol',
    patient_drugs         = ['lisinopril', 'metformin'],
    patient_comorbidities = [],
    n_mc_passes           = 15
)

# Simulate what the backend adds for a private drug
private_drug_result = generate_full_card(
    drug_name             = 'tramadol',
    patient_drugs         = ['sertraline'],
    patient_comorbidities = [],
    n_mc_passes           = 15
)

# Apply privacy mask — drug name hidden, warning shown
if private_drug_result.get('interactions'):
    private_card = private_drug_result['interactions'][0].copy()
    private_card['existing_drug']        = '*** PRIVATE MEDICATION ***'
    private_card['existing_ingredient']  = '*** PRIVATE MEDICATION ***'
    private_card['mechanism']            = 'Conflict detected with a restricted medication. Contact patient before prescribing.'
    private_card['side_effect']          = 'Hidden — medication is private'
    private_card['privacy_note']         = 'This drug has been marked private by the patient. The interaction check has been performed but the drug identity cannot be revealed.'
    private_card['alert_badge']          = private_drug_result['overall_badge']

    print(f"  Regular drugs badge  : [{result4['overall_badge']}]")
    print(f"  Private drug badge   : [{private_card['alert_badge']}]")
    print(f"  Existing drug shown  : {private_card['existing_drug']}")
    print(f"  Privacy note         : {private_card['privacy_note'][:80]}...")


# ── STEP 7: SAVE CARD ENGINE ──────────────────────────────────

print("\n[STEP 5/7] Saving card engine outputs...")

# Save all test results
all_test_results = {
    'test1_simple'        : result1,
    'test2_comorbidity'   : result2,
    'test3_multi_ingredient': result3,
}

with open('output/cards/test_card_results.json', 'w') as f:
    json.dump(all_test_results, f, indent=2)

# Save the composition dictionary for backend use
with open('output/cards/drug_compositions.json', 'w') as f:
    json.dump(VALIDATED_COMPOSITIONS, f, indent=2)

# Save comorbidity profiles for backend use
with open('output/cards/comorbidity_profiles.json', 'w') as f:
    json.dump(COMORBIDITY_RISK_PROFILES, f, indent=2)

# Save the card generator config
card_engine_config = {
    'risk_threshold'     : RISK_THRESHOLD,
    'uncertainty_threshold': UNCERT_THRESH,
    'n_mc_passes'        : N_PASSES,
    'multi_ingredient_drugs': len(VALIDATED_COMPOSITIONS),
    'comorbidity_profiles'  : len(COMORBIDITY_RISK_PROFILES),
    'drug_vocabulary_size'  : len(drug_to_id),
    'interaction_pairs'     : len(interaction_lookup),
    'twosides_pairs'        : len(twosides_lookup),
}

with open('output/cards/card_engine_config.json', 'w') as f:
    json.dump(card_engine_config, f, indent=2)

print(f"  output/cards/test_card_results.json")
print(f"  output/cards/drug_compositions.json")
print(f"  output/cards/comorbidity_profiles.json")
print(f"  output/cards/card_engine_config.json")

print(f"\n  Card engine config:")
for k, v in card_engine_config.items():
    print(f"    {k:<30} : {v}")

print(f"\n{'=' * 65}")
print(f"  Phase 2 | Week 6 COMPLETE")
print(f"  NEXT: Week 7 — privacy_shield.py (Patent 3)")
print(f"{'=' * 65}")