# ================================================================
#  PolyGuard — Phase 2, Week 8
#  consent_layer.py — Patient Consent Management System
#  Integrates: Privacy Shield + Card Engine + Consent DB
#  Status: Week 8 / 20
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
import pickle
import json
import os
import uuid
from datetime import datetime
from enum import Enum

os.makedirs('output/consent', exist_ok=True)

print("=" * 65)
print("  PolyGuard — Patient Consent Management System")
print("  Phase 2 | Week 8 | Step 8 of 20")
print("=" * 65)

# ── ENUMS ─────────────────────────────────────────────────────

class PrivacyLevel(Enum):
    FULL_SHARE       = "full_share"
    INTERACTION_ONLY = "interaction_only"
    FULL_PRIVATE     = "full_private"

class AlertBadge(Enum):
    RED    = "RED"
    YELLOW = "YELLOW"
    GREEN  = "GREEN"


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
        z = self.encode(x, edge_index)
        if private_node_ids:
            mask = torch.ones_like(z)
            for node_id in private_node_ids:
                mask[node_id] = 0.0
            z = z * mask
        return z

    def forward_with_privacy(self, x, edge_index,
                             pred_edges, private_node_ids=None):
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

print("\n[STEP 1/6] Loading all resources...")

graph = torch.load('output/polyguard_graph.pt', weights_only=False)

with open('output/drug_to_id.pkl', 'rb') as f:
    drug_to_id = pickle.load(f)
with open('output/id_to_drug.pkl', 'rb') as f:
    id_to_drug = pickle.load(f)
with open('output/twosides_lookup.pkl', 'rb') as f:
    twosides_lookup = pickle.load(f)
with open('output/cards/drug_compositions.json') as f:
    drug_compositions = json.load(f)
with open('output/cards/comorbidity_profiles.json') as f:
    comorbidity_profiles = json.load(f)

bayesian_config = torch.load(
    'output/bayesian/bayesian_config.pt', weights_only=True
)
RISK_THRESHOLD = bayesian_config['risk_threshold']
UNCERT_THRESH  = bayesian_config['optimal_threshold']

import pandas as pd
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
num_edges   = graph.edge_index.shape[1]
perm        = torch.randperm(num_edges,
              generator=torch.Generator().manual_seed(42))
train_edges = graph.edge_index[:, perm][:, :int(num_edges * 0.80)]

print(f"  Model loaded        : OK")
print(f"  Drug vocabulary     : {len(drug_to_id):,}")
print(f"  Interaction texts   : {len(interaction_lookup):,}")


# ══════════════════════════════════════════════════════════════
# CONSENT DATABASE MANAGER
# Simulates PostgreSQL in Week 9 — same interface, JSON backend.
# FastAPI backend swaps this for SQLAlchemy in Week 10.
# ══════════════════════════════════════════════════════════════

class ConsentDatabase:
    """
    Manages patient consent flags and medication profiles.
    JSON-backed for Phase 2 testing.
    Replaced by PostgreSQL in Phase 3 (Week 9).
    Interface kept identical so backend swap is seamless.
    """

    def __init__(self, db_path='output/consent/consent_db.json'):
        self.db_path = db_path
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path) as f:
                self.db = json.load(f)
        else:
            self.db = {
                'patients'     : {},
                'doctors'      : {},
                'override_logs': [],
            }

    def _save(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.db, f, indent=2)

    # ── DOCTOR MANAGEMENT ─────────────────────────────────────

    def register_doctor(self, name, specialty):
        """Doctor account creation."""
        doctor_id = f"DR{str(uuid.uuid4())[:6].upper()}"
        self.db['doctors'][doctor_id] = {
            'doctor_id' : doctor_id,
            'name'      : name,
            'specialty' : specialty,
            'created_at': datetime.now().isoformat(),
        }
        self._save()
        return doctor_id

    def get_doctor(self, doctor_id):
        return self.db['doctors'].get(doctor_id)

    # ── PATIENT MANAGEMENT ────────────────────────────────────

    def register_patient(self, name, phone, age,
                         comorbidities, created_by_doctor_id):
        """
        Doctor creates patient account as agreed in project plan.
        Returns patient_id and temporary password for patient login.
        """
        patient_id   = f"PT{str(uuid.uuid4())[:6].upper()}"
        temp_password = f"poly{str(uuid.uuid4())[:4].upper()}"

        self.db['patients'][patient_id] = {
            'patient_id'          : patient_id,
            'name'                : name,
            'phone'               : phone,
            'age'                 : age,
            'comorbidities'       : comorbidities,
            'medications'         : [],
            'created_by'          : created_by_doctor_id,
            'created_at'          : datetime.now().isoformat(),
            'temp_password'       : temp_password,
            'password_changed'    : False,
        }
        self._save()
        return patient_id, temp_password

    def get_patient(self, patient_id):
        return self.db['patients'].get(patient_id)

    def get_patient_by_phone(self, phone):
        """Doctor searches patient by phone number."""
        for pid, patient in self.db['patients'].items():
            if patient['phone'] == phone:
                return patient
        return None

    def update_comorbidities(self, patient_id,
                             comorbidities, updated_by_doctor_id):
        """Only doctors can update comorbidities — agreed design."""
        if patient_id not in self.db['patients']:
            return False
        self.db['patients'][patient_id]['comorbidities'] = comorbidities
        self.db['patients'][patient_id]['comorbidities_updated_by'] = updated_by_doctor_id
        self.db['patients'][patient_id]['comorbidities_updated_at'] = datetime.now().isoformat()
        self._save()
        return True

    # ── MEDICATION MANAGEMENT ─────────────────────────────────

    def add_medication(self, patient_id, drug_name,
                       dose, prescribed_by_doctor_id,
                       privacy_level=PrivacyLevel.FULL_SHARE,
                       override_reason=None):
        """
        Doctor adds a drug to patient profile.
        Includes override reason if RED alert was bypassed.
        """
        if patient_id not in self.db['patients']:
            return False

        med_entry = {
            'med_id'              : str(uuid.uuid4())[:8],
            'drug_name'           : drug_name.strip().lower(),
            'drug_display'        : drug_name.title(),
            'dose'                : dose,
            'privacy_level'       : privacy_level.value,
            'prescribed_by'       : prescribed_by_doctor_id,
            'prescribed_at'       : datetime.now().isoformat(),
            'active'              : True,
            'override_reason'     : override_reason,
        }

        self.db['patients'][patient_id]['medications'].append(med_entry)
        self._save()

        # Log override if reason provided
        if override_reason:
            self.log_override(
                patient_id, drug_name,
                prescribed_by_doctor_id, override_reason
            )

        return med_entry['med_id']

    def update_privacy_level(self, patient_id, drug_name, new_level):
        """
        Patient updates their own privacy setting per drug.
        Only privacy level can be changed by patient — not dose or drug.
        """
        if patient_id not in self.db['patients']:
            return False
        for med in self.db['patients'][patient_id]['medications']:
            if med['drug_name'] == drug_name.strip().lower():
                med['privacy_level']    = new_level.value
                med['privacy_updated_at'] = datetime.now().isoformat()
                self._save()
                return True
        return False

    def get_medication_profile(self, patient_id,
                               requesting_doctor_id=None):
        """
        Returns patient medication list with privacy levels.
        For doctors — drugs marked INTERACTION_ONLY shown as masked.
        For patient themselves — all drugs shown.
        """
        patient = self.db['patients'].get(patient_id)
        if not patient:
            return None

        meds = []
        for med in patient['medications']:
            if not med['active']:
                continue

            privacy = PrivacyLevel(med['privacy_level'])

            # Patient viewing their own profile — show everything
            if requesting_doctor_id is None:
                meds.append({
                    **med,
                    'visible_to_doctor': True,
                })

            # Doctor viewing — apply privacy rules
            else:
                if privacy == PrivacyLevel.FULL_SHARE:
                    meds.append({
                        **med,
                        'visible_to_doctor': True,
                    })
                elif privacy == PrivacyLevel.INTERACTION_ONLY:
                    # Drug exists in interaction check but name hidden
                    meds.append({
                        'med_id'            : med['med_id'],
                        'drug_name'         : '*** PRIVATE ***',
                        'drug_display'      : '*** PRIVATE MEDICATION ***',
                        'dose'              : '*** HIDDEN ***',
                        'privacy_level'     : med['privacy_level'],
                        'prescribed_by'     : '*** HIDDEN ***',
                        'active'            : True,
                        'visible_to_doctor' : False,
                        'interaction_check' : True,
                    })
                elif privacy == PrivacyLevel.FULL_PRIVATE:
                    # Drug not shown and not checked
                    meds.append({
                        'med_id'            : med['med_id'],
                        'drug_name'         : '*** PRIVATE ***',
                        'drug_display'      : '*** FULLY PRIVATE — NOT CHECKED ***',
                        'privacy_level'     : med['privacy_level'],
                        'active'            : True,
                        'visible_to_doctor' : False,
                        'interaction_check' : False,
                    })

        return meds

    # ── OVERRIDE LOGGING ──────────────────────────────────────

    def log_override(self, patient_id, drug_name,
                     doctor_id, reason):
        """
        Logs doctor override when RED alert is bypassed.
        Agreed design — system helps doctors, does not block them.
        """
        log_entry = {
            'log_id'     : str(uuid.uuid4())[:8],
            'patient_id' : patient_id,
            'drug_name'  : drug_name,
            'doctor_id'  : doctor_id,
            'reason'     : reason,
            'timestamp'  : datetime.now().isoformat(),
        }
        self.db['override_logs'].append(log_entry)
        self._save()
        return log_entry['log_id']

    def get_override_logs(self, patient_id=None):
        logs = self.db['override_logs']
        if patient_id:
            logs = [l for l in logs if l['patient_id'] == patient_id]
        return logs


# ══════════════════════════════════════════════════════════════
# MASTER INTERACTION CHECKER
# This is the single function FastAPI calls in Week 10.
# Combines: consent DB + privacy shield + card engine + Bayesian
# ══════════════════════════════════════════════════════════════

def get_ingredients(drug_name):
    name = drug_name.strip().lower()
    if name in drug_compositions:
        return drug_compositions[name]
    return [name] if name in drug_to_id else []


def mc_predict(d1, d2, private_ids=None, n_passes=15):
    if d1 not in drug_to_id or d2 not in drug_to_id:
        return None, None
    pred_edge = torch.tensor(
        [[drug_to_id[d1]], [drug_to_id[d2]]], dtype=torch.long
    )
    private_ids = private_ids or []
    model.train()
    probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            score, _ = model.forward_with_privacy(
                graph.x, train_edges, pred_edge, private_ids
            )
            probs.append(torch.sigmoid(score).item())
    return float(np.mean(probs)), float(np.std(probs))


def classify_severity(text):
    if not text:
        return 'MODERATE'
    text = text.lower()
    for kw in ['severe','fatal','life-threatening','toxicity',
                'seizure','death','hemorrhage','anaphylaxis']:
        if kw in text:
            return 'SEVERE'
    return 'MODERATE'


def get_badge(prob, uncert, severity):
    risky     = prob   > RISK_THRESHOLD
    confident = uncert < UNCERT_THRESH
    if risky and confident:
        return 'RED'
    elif risky:
        return 'YELLOW'
    return 'GREEN'


def check_comorbidity(text, comorbidities):
    warnings = []
    if not text or not comorbidities:
        return warnings
    text_l = text.lower()
    for cond in comorbidities:
        key     = cond.lower().replace(' ', '_').replace('-', '_')
        profile = comorbidity_profiles.get(key)
        if not profile:
            for k, p in comorbidity_profiles.items():
                if k in key or key in k:
                    profile = p
                    break
        if profile:
            for kw in profile['risk_keywords']:
                if kw in text_l:
                    warnings.append({
                        'condition'      : profile['display_name'],
                        'warning'        : profile['warning'],
                        'risk_multiplier': profile['risk_multiplier'],
                    })
                    break
    return warnings


def master_check_interaction(
    new_drug,
    patient_id,
    consent_db,
    requesting_doctor_id,
    n_mc_passes=15
):
    """
    MASTER FUNCTION — called by FastAPI /check-interaction endpoint.

    1. Loads patient profile from consent database
    2. Decomposes multi-ingredient drugs
    3. Applies privacy shield per drug privacy level
    4. Runs Bayesian MC Dropout prediction
    5. Applies comorbidity risk adjustment
    6. Returns complete card set for doctor dashboard
    """
    patient = consent_db.get_patient(patient_id)
    if not patient:
        return {'status': 'error', 'message': 'Patient not found'}

    comorbidities = patient.get('comorbidities', [])
    new_ingredients = get_ingredients(new_drug)
    if not new_ingredients:
        return {
            'status' : 'error',
            'message': f"'{new_drug}' not found in drug vocabulary",
        }

    all_cards       = []
    blind_spots     = 0
    has_blind_spot  = False

    for med in patient['medications']:
        if not med['active']:
            continue

        existing_drug = med['drug_name']
        privacy       = PrivacyLevel(med['privacy_level'])

        # Full private — skip check, note blind spot
        if privacy == PrivacyLevel.FULL_PRIVATE:
            blind_spots += 1
            has_blind_spot = True
            continue

        existing_ingredients = get_ingredients(existing_drug)

        for new_ing in new_ingredients:
            for exist_ing in existing_ingredients:
                if new_ing == exist_ing:
                    continue

                # Apply privacy mask if interaction_only
                private_ids = []
                if privacy == PrivacyLevel.INTERACTION_ONLY:
                    if exist_ing in drug_to_id:
                        private_ids = [drug_to_id[exist_ing]]

                prob, uncert = mc_predict(
                    new_ing, exist_ing, private_ids, n_mc_passes
                )
                if prob is None:
                    continue

                mechanism = interaction_lookup.get(
                    (new_ing, exist_ing),
                    interaction_lookup.get(
                        (exist_ing, new_ing), None
                    )
                )
                side_effect = twosides_lookup.get(
                    (new_ing, exist_ing),
                    twosides_lookup.get(
                        (exist_ing, new_ing), 'Not available'
                    )
                )

                severity = classify_severity(mechanism)

                # Comorbidity adjustment (CC-DGNN Patent 1)
                comorbidity_warnings = check_comorbidity(
                    mechanism, comorbidities
                )
                adjusted_prob = prob
                if comorbidity_warnings:
                    mx = max(
                        w['risk_multiplier']
                        for w in comorbidity_warnings
                    )
                    adjusted_prob = min(prob * mx, 1.0)

                badge = get_badge(adjusted_prob, uncert, severity)

                # Build card — apply privacy masking to display
                if privacy == PrivacyLevel.INTERACTION_ONLY:
                    card = {
                        'new_drug'           : new_drug.title(),
                        'existing_drug'      : '*** PRIVATE MEDICATION ***',
                        'alert_badge'        : badge,
                        'risk_score'         : round(prob, 4),
                        'uncertainty'        : round(uncert, 4),
                        'severity'           : severity,
                        'mechanism'          : (
                            'Conflict detected with a restricted medication. '
                            'Contact patient before prescribing.'
                        ),
                        'side_effect'        : 'Hidden — medication is private',
                        'privacy_level'      : privacy.value,
                        'privacy_note'       : (
                            'This drug is marked private. '
                            'Interaction check performed — identity hidden.'
                        ),
                        'comorbidity_warnings': [],
                        'override_allowed'   : True,
                        'override_required_reason': badge in ['RED','YELLOW'],
                        'clinical_action'    : (
                            'CONTACT PATIENT — conflict with restricted medication'
                            if badge == 'RED' else
                            'VERIFY with patient — possible conflict'
                            if badge == 'YELLOW' else
                            'No conflict with restricted medication'
                        ),
                        'is_multi_ingredient': len(new_ingredients) > 1,
                    }
                else:
                    card = {
                        'new_drug'           : new_drug.title(),
                        'existing_drug'      : existing_drug.title(),
                        'new_ingredient'     : new_ing.title(),
                        'existing_ingredient': exist_ing.title(),
                        'alert_badge'        : badge,
                        'risk_score'         : round(prob, 4),
                        'adjusted_risk_score': round(adjusted_prob, 4),
                        'uncertainty'        : round(uncert, 4),
                        'confidence'         : 'High' if uncert < UNCERT_THRESH else 'Low',
                        'severity'           : severity,
                        'mechanism'          : (
                            mechanism[:300]
                            if mechanism else 'See clinical guidelines'
                        ),
                        'side_effect'        : side_effect,
                        'privacy_level'      : privacy.value,
                        'comorbidity_warnings': comorbidity_warnings,
                        'comorbidity_adjusted': len(comorbidity_warnings) > 0,
                        'override_allowed'   : True,
                        'override_required_reason': badge in ['RED','YELLOW'],
                        'clinical_action'    : (
                            'DO NOT PRESCRIBE without specialist consultation'
                            if badge == 'RED' else
                            'Prescribe with caution — monitor patient closely'
                            if badge == 'YELLOW' else
                            'Safe to prescribe — routine monitoring advised'
                        ),
                        'is_multi_ingredient': len(new_ingredients) > 1,
                    }

                all_cards.append(card)

    # Sort by risk score descending
    all_cards.sort(
        key=lambda c: c.get('adjusted_risk_score',
                            c.get('risk_score', 0)),
        reverse=True
    )

    badges  = [c['alert_badge'] for c in all_cards]
    overall = ('RED' if 'RED' in badges else
               'YELLOW' if 'YELLOW' in badges else 'GREEN')

    return {
        'status'          : 'ok',
        'patient_id'      : patient_id,
        'new_drug'        : new_drug.title(),
        'overall_badge'   : overall,
        'cards'           : all_cards,
        'summary'         : {
            'total_checks' : len(all_cards),
            'red_count'    : badges.count('RED'),
            'yellow_count' : badges.count('YELLOW'),
            'green_count'  : badges.count('GREEN'),
            'blind_spots'  : blind_spots,
        },
        'blind_spot_warning': (
            'WARNING: Interaction check is INCOMPLETE. '
            f'{blind_spots} medication(s) are fully private. '
            'Consult patient directly.'
        ) if has_blind_spot else None,
        'comorbidities'   : comorbidities,
        'override_allowed': True,
    }


# ── STEP 2: UNIT TESTS ────────────────────────────────────────

print("\n[STEP 2/6] Running unit tests...\n")

db = ConsentDatabase()

# ── Setup test data ───────────────────────────────────────────
dr_id = db.register_doctor('Dr. Asha N', 'Clinical Pharmacology')
pt_id, temp_pw = db.register_patient(
    name              = 'John Patient',
    phone             = '+91-9876543210',
    age               = 72,
    comorbidities     = ['kidney_failure', 'heart_failure'],
    created_by_doctor_id = dr_id
)

# Add medications with different privacy levels
db.add_medication(pt_id, 'warfarin',   '5mg',  dr_id, PrivacyLevel.FULL_SHARE)
db.add_medication(pt_id, 'sertraline', '50mg', dr_id, PrivacyLevel.INTERACTION_ONLY)
db.add_medication(pt_id, 'metformin',  '500mg',dr_id, PrivacyLevel.FULL_SHARE)
db.add_medication(pt_id, 'alprazolam', '0.5mg',dr_id, PrivacyLevel.FULL_PRIVATE)

print(f"  Test patient created  : {pt_id}")
print(f"  Temp password         : {temp_pw}")
print(f"  Doctor ID             : {dr_id}")
print(f"  Medications added     : 4")
print()

# ── TEST 1 — Patient search by phone ─────────────────────────
print("  TEST 1 — Doctor searches patient by phone number")
found = db.get_patient_by_phone('+91-9876543210')
assert found is not None, "Patient not found by phone"
assert found['patient_id'] == pt_id, "Wrong patient returned"
print(f"  PASS — Patient found: {found['name']} ({found['patient_id']})")

# ── TEST 2 — Doctor view respects privacy flags ───────────────
print("\n  TEST 2 — Doctor view applies privacy rules correctly")
doctor_view = db.get_medication_profile(pt_id, requesting_doctor_id=dr_id)
for med in doctor_view:
    priv = med['privacy_level']
    if priv == 'interaction_only':
        assert med['drug_name'] == '*** PRIVATE ***', \
            "Interaction-only drug should be masked"
    elif priv == 'full_private':
        assert 'FULLY PRIVATE' in med['drug_display'], \
            "Full private drug should show private label"
    elif priv == 'full_share':
        assert med['drug_name'] not in ['*** PRIVATE ***'], \
            "Full share drug should be visible"
print(f"  PASS — Privacy rules applied correctly to all 4 medications")

# ── TEST 3 — Patient view shows all drugs ─────────────────────
print("\n  TEST 3 — Patient sees all their own medications")
patient_view = db.get_medication_profile(pt_id, requesting_doctor_id=None)
visible = [m for m in patient_view if '*** PRIVATE ***' not in m['drug_name']]
assert len(visible) == 4, f"Patient should see all 4 drugs, got {len(visible)}"
print(f"  PASS — Patient sees all 4 medications including private ones")

# ── TEST 4 — Patient updates privacy level ────────────────────
print("\n  TEST 4 — Patient changes privacy level for a drug")
result = db.update_privacy_level(pt_id, 'sertraline', PrivacyLevel.FULL_SHARE)
assert result == True, "Privacy update failed"
updated = db.get_patient(pt_id)
for med in updated['medications']:
    if med['drug_name'] == 'sertraline':
        assert med['privacy_level'] == 'full_share', \
            "Privacy level not updated"
print(f"  PASS — Sertraline privacy changed to FULL_SHARE successfully")
# Restore for further tests
db.update_privacy_level(pt_id, 'sertraline', PrivacyLevel.INTERACTION_ONLY)

# ── TEST 5 — Doctor override logging ─────────────────────────
print("\n  TEST 5 — Doctor override on RED alert is logged")
db.add_medication(
    pt_id, 'tramadol', '50mg', dr_id,
    PrivacyLevel.FULL_SHARE,
    override_reason='Patient in severe post-operative pain. Risk accepted after specialist consultation.'
)
logs = db.get_override_logs(patient_id=pt_id)
assert len(logs) >= 1, "Override not logged"
assert logs[-1]['reason'] != '', "Override reason not saved"
print(f"  PASS — Override logged: '{logs[-1]['reason'][:60]}...'")

# ── TEST 6 — Master interaction checker ──────────────────────
print("\n  TEST 6 — Master interaction checker (full system test)")
print("  Adding Ciprofloxacin to patient profile...")

check_result = master_check_interaction(
    new_drug             = 'ciprofloxacin',
    patient_id           = pt_id,
    consent_db           = db,
    requesting_doctor_id = dr_id,
    n_mc_passes          = 10
)

assert check_result['status'] == 'ok', "Master check failed"
print(f"\n  Overall badge        : [{check_result['overall_badge']}]")
print(f"  Patient              : {check_result['patient_id']}")
print(f"  Comorbidities        : {check_result['comorbidities']}")
print(f"  Summary              : {check_result['summary']}")
if check_result['blind_spot_warning']:
    print(f"  ⚠ {check_result['blind_spot_warning']}")
print(f"\n  Top interaction cards:")
for card in check_result['cards'][:4]:
    print(f"    {card['new_drug']:<18} + "
          f"{card['existing_drug']:<30} "
          f"[{card['alert_badge']}] "
          f"score:{card.get('adjusted_risk_score', card.get('risk_score','?'))}")
    if card.get('comorbidity_warnings'):
        print(f"      ⚠ Comorbidity: {card['comorbidity_warnings'][0]['condition']}")
print(f"\n  PASS — Master interaction checker working correctly")


# ── STEP 3: SAVE CONSENT LAYER OUTPUTS ───────────────────────

print("\n\n[STEP 3/6] Saving consent layer outputs...")

# Save the full test result
with open('output/consent/master_check_test.json', 'w') as f:
    json.dump(check_result, f, indent=2)

# Save unit test summary
unit_test_summary = {
    'tests_run'    : 6,
    'tests_passed' : 6,
    'tests_failed' : 0,
    'test_names'   : [
        'Patient search by phone number',
        'Doctor view applies privacy rules',
        'Patient sees all own medications',
        'Patient updates privacy level',
        'Doctor override logging',
        'Master interaction checker full system test',
    ],
    'patient_id'   : pt_id,
    'doctor_id'    : dr_id,
}

with open('output/consent/unit_test_summary.json', 'w') as f:
    json.dump(unit_test_summary, f, indent=2)

print(f"  output/consent/consent_db.json          — patient and doctor data")
print(f"  output/consent/master_check_test.json   — full system test result")
print(f"  output/consent/unit_test_summary.json   — all 6 unit tests")
print(f"\n  Unit test summary:")
print(f"  Tests run    : {unit_test_summary['tests_run']}")
print(f"  Tests passed : {unit_test_summary['tests_passed']}")
print(f"  Tests failed : {unit_test_summary['tests_failed']}")

print(f"\n{'=' * 65}")
print(f"  Phase 2 | Week 8 COMPLETE")
print(f"  Phase 2 is now FULLY COMPLETE")
print(f"  NEXT: Phase 3 | Week 9 — Database setup (PostgreSQL)")
print(f"{'=' * 65}")