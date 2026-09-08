# ================================================================
#  PolyGuard — Phase 3, Week 9
#  database_setup.py — PostgreSQL Schema + Seed Data
#  Status: Week 9 / 20
# ================================================================

import psycopg2
from psycopg2.extras import RealDictCursor
import json
import os
import uuid
import hashlib
from datetime import datetime

print("=" * 65)
print("  PolyGuard — Database Setup")
print("  Phase 3 | Week 9 | Step 9 of 20")
print("=" * 65)

# ── DATABASE CONNECTION ───────────────────────────────────────

DB_CONFIG = {
    'host'    : 'localhost',
    'port'    : 5432,
    'database': 'polyguard',
    'user'    : 'postgres',
    'password': 'polyguard123',
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ── STEP 1: CREATE ALL TABLES ─────────────────────────────────

print("\n[STEP 1/6] Creating database tables...")

conn = get_connection()
cur  = conn.cursor()

# Drop existing tables for clean setup (development only)
cur.execute("""
    DROP TABLE IF EXISTS interaction_cache  CASCADE;
    DROP TABLE IF EXISTS override_logs      CASCADE;
    DROP TABLE IF EXISTS consent_history    CASCADE;
    DROP TABLE IF EXISTS medications        CASCADE;
    DROP TABLE IF EXISTS drug_compositions  CASCADE;
    DROP TABLE IF EXISTS patients           CASCADE;
    DROP TABLE IF EXISTS doctors            CASCADE;
    DROP TABLE IF EXISTS comorbidity_profiles CASCADE;
""")

# ── TABLE 1: DOCTORS ──────────────────────────────────────────
cur.execute("""
    CREATE TABLE doctors (
        doctor_id       VARCHAR(20) PRIMARY KEY,
        name            VARCHAR(100) NOT NULL,
        specialty       VARCHAR(100),
        email           VARCHAR(100) UNIQUE,
        phone           VARCHAR(20),
        password_hash   VARCHAR(64) NOT NULL,
        is_active       BOOLEAN DEFAULT TRUE,
        created_at      TIMESTAMP DEFAULT NOW()
    );
""")
print("  Created table: doctors")

# ── TABLE 2: PATIENTS ─────────────────────────────────────────
cur.execute("""
    CREATE TABLE patients (
        patient_id          VARCHAR(20) PRIMARY KEY,
        name                VARCHAR(100) NOT NULL,
        phone               VARCHAR(20) UNIQUE NOT NULL,
        date_of_birth       DATE,
        age                 INTEGER,
        blood_group         VARCHAR(5),
        comorbidities       TEXT[],
        created_by_doctor   VARCHAR(20) REFERENCES doctors(doctor_id),
        password_hash       VARCHAR(64) NOT NULL,
        password_changed    BOOLEAN DEFAULT FALSE,
        is_active           BOOLEAN DEFAULT TRUE,
        created_at          TIMESTAMP DEFAULT NOW()
    );
""")
print("  Created table: patients")

# ── TABLE 3: MEDICATIONS ──────────────────────────────────────
cur.execute("""
    CREATE TABLE medications (
        med_id              VARCHAR(20) PRIMARY KEY,
        patient_id          VARCHAR(20) REFERENCES patients(patient_id),
        drug_name           VARCHAR(100) NOT NULL,
        drug_display        VARCHAR(100),
        dose                VARCHAR(50),
        frequency           VARCHAR(50),
        prescribed_by       VARCHAR(20) REFERENCES doctors(doctor_id),
        privacy_level       VARCHAR(20) DEFAULT 'full_share'
                            CHECK (privacy_level IN
                            ('full_share','interaction_only','full_private')),
        is_active           BOOLEAN DEFAULT TRUE,
        override_reason     TEXT,
        prescribed_at       TIMESTAMP DEFAULT NOW(),
        discontinued_at     TIMESTAMP,
        discontinued_by     VARCHAR(20)
    );
    CREATE INDEX idx_medications_patient
        ON medications(patient_id);
    CREATE INDEX idx_medications_privacy
        ON medications(privacy_level);
""")
print("  Created table: medications")

# ── TABLE 4: DRUG COMPOSITIONS ────────────────────────────────
# Multi-ingredient tablet support — agreed in project design
cur.execute("""
    CREATE TABLE drug_compositions (
        composition_id  SERIAL PRIMARY KEY,
        tablet_name     VARCHAR(100) NOT NULL,
        ingredient      VARCHAR(100) NOT NULL,
        ingredient_order INTEGER DEFAULT 1,
        created_at      TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX idx_compositions_tablet
        ON drug_compositions(tablet_name);
""")
print("  Created table: drug_compositions")

# ── TABLE 5: OVERRIDE LOGS ────────────────────────────────────
# Logs every RED/YELLOW alert override with doctor's reason
# Agreed design — system never blocks doctors
cur.execute("""
    CREATE TABLE override_logs (
        log_id          VARCHAR(20) PRIMARY KEY,
        patient_id      VARCHAR(20) REFERENCES patients(patient_id),
        drug_name       VARCHAR(100),
        alert_badge     VARCHAR(10),
        risk_score      DECIMAL(6,4),
        doctor_id       VARCHAR(20) REFERENCES doctors(doctor_id),
        override_reason TEXT NOT NULL,
        created_at      TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX idx_override_patient
        ON override_logs(patient_id);
""")
print("  Created table: override_logs")

# ── TABLE 6: CONSENT HISTORY ──────────────────────────────────
# Tracks every privacy level change per drug per patient
cur.execute("""
    CREATE TABLE consent_history (
        history_id      SERIAL PRIMARY KEY,
        patient_id      VARCHAR(20) REFERENCES patients(patient_id),
        med_id          VARCHAR(20) REFERENCES medications(med_id),
        drug_name       VARCHAR(100),
        old_privacy     VARCHAR(20),
        new_privacy     VARCHAR(20),
        changed_by      VARCHAR(20),
        changed_at      TIMESTAMP DEFAULT NOW()
    );
""")
print("  Created table: consent_history")

# ── TABLE 7: INTERACTION CACHE ────────────────────────────────
# Caches recent GNN predictions for speed
# Same drug pair = same result within 24 hours
cur.execute("""
    CREATE TABLE interaction_cache (
        cache_id        SERIAL PRIMARY KEY,
        drug1           VARCHAR(100) NOT NULL,
        drug2           VARCHAR(100) NOT NULL,
        risk_score      DECIMAL(6,4),
        uncertainty     DECIMAL(6,4),
        alert_badge     VARCHAR(10),
        mechanism       TEXT,
        side_effect     TEXT,
        created_at      TIMESTAMP DEFAULT NOW(),
        expires_at      TIMESTAMP DEFAULT NOW() + INTERVAL '24 hours',
        UNIQUE(drug1, drug2)
    );
    CREATE INDEX idx_cache_drugs
        ON interaction_cache(drug1, drug2);
""")
print("  Created table: interaction_cache")

# ── TABLE 8: COMORBIDITY PROFILES ─────────────────────────────
cur.execute("""
    CREATE TABLE comorbidity_profiles (
        profile_id      SERIAL PRIMARY KEY,
        condition_key   VARCHAR(50) UNIQUE NOT NULL,
        display_name    VARCHAR(100),
        risk_keywords   TEXT[],
        warning_text    TEXT,
        risk_multiplier DECIMAL(4,2) DEFAULT 1.0
    );
""")
print("  Created table: comorbidity_profiles")

conn.commit()
print("\n  All 8 tables created successfully")


# ── STEP 2: SEED DRUG COMPOSITIONS ───────────────────────────

print("\n[STEP 2/6] Seeding drug compositions...")

with open('output/cards/drug_compositions.json') as f:
    drug_compositions = json.load(f)

for tablet_name, ingredients in drug_compositions.items():
    for i, ingredient in enumerate(ingredients):
        cur.execute("""
            INSERT INTO drug_compositions
                (tablet_name, ingredient, ingredient_order)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (tablet_name, ingredient, i + 1))

conn.commit()
print(f"  Seeded {len(drug_compositions)} multi-ingredient drugs")

# Sample check
cur.execute("""
    SELECT tablet_name, ingredient, ingredient_order
    FROM drug_compositions
    WHERE tablet_name = 'co-amoxiclav'
    ORDER BY ingredient_order
""")
rows = cur.fetchall()
print(f"  Sample — co-amoxiclav: {[r[1] for r in rows]}")


# ── STEP 3: SEED COMORBIDITY PROFILES ────────────────────────

print("\n[STEP 3/6] Seeding comorbidity profiles...")

with open('output/cards/comorbidity_profiles.json') as f:
    comorbidity_data = json.load(f)

for key, profile in comorbidity_data.items():
    cur.execute("""
        INSERT INTO comorbidity_profiles
            (condition_key, display_name, risk_keywords,
             warning_text, risk_multiplier)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (condition_key) DO NOTHING
    """, (
        key,
        profile['display_name'],
        profile['risk_keywords'],
        profile['warning'],
        profile['risk_multiplier'],
    ))

conn.commit()
print(f"  Seeded {len(comorbidity_data)} comorbidity profiles")

cur.execute("SELECT condition_key, risk_multiplier FROM comorbidity_profiles")
profiles = cur.fetchall()
for p in profiles:
    print(f"  {p[0]:<25} multiplier: {p[1]}")


# ── STEP 4: SEED DEMO DATA ────────────────────────────────────

print("\n[STEP 4/6] Seeding demo data...")

# Create demo doctor
demo_doctor_id = "DR000001"
cur.execute("""
    INSERT INTO doctors
        (doctor_id, name, specialty, email,
         phone, password_hash)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (doctor_id) DO NOTHING
""", (
    demo_doctor_id,
    'Dr. Asha N',
    'Clinical Pharmacology',
    'asha.n@vit.ac.in',
    '+91-9000000001',
    hash_password('doctor123'),
))

# Create demo patient
demo_patient_id = "PT000001"
cur.execute("""
    INSERT INTO patients
        (patient_id, name, phone, age,
         comorbidities, created_by_doctor,
         password_hash)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (patient_id) DO NOTHING
""", (
    demo_patient_id,
    'John Demo Patient',
    '+91-9876543210',
    72,
    ['kidney_failure', 'heart_failure'],
    demo_doctor_id,
    hash_password('patient123'),
))

# Add demo medications
demo_meds = [
    ('MED00001', demo_patient_id, 'warfarin',    'Warfarin',    '5mg',  'Once daily',   demo_doctor_id, 'full_share'),
    ('MED00002', demo_patient_id, 'sertraline',  'Sertraline',  '50mg', 'Once daily',   demo_doctor_id, 'interaction_only'),
    ('MED00003', demo_patient_id, 'metformin',   'Metformin',   '500mg','Twice daily',  demo_doctor_id, 'full_share'),
    ('MED00004', demo_patient_id, 'alprazolam',  'Alprazolam',  '0.5mg','As needed',    demo_doctor_id, 'full_private'),
    ('MED00005', demo_patient_id, 'lisinopril',  'Lisinopril',  '10mg', 'Once daily',   demo_doctor_id, 'full_share'),
]

for med in demo_meds:
    cur.execute("""
        INSERT INTO medications
            (med_id, patient_id, drug_name, drug_display,
             dose, frequency, prescribed_by, privacy_level)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (med_id) DO NOTHING
    """, med)

conn.commit()
print(f"  Demo doctor created   : {demo_doctor_id} — Dr. Asha N")
print(f"  Demo patient created  : {demo_patient_id} — John Demo Patient")
print(f"  Demo medications added: {len(demo_meds)}")


# ── STEP 5: DATABASE MANAGER CLASS ───────────────────────────

print("\n[STEP 5/6] Creating database manager...")

class PolyGuardDB:
    """
    PostgreSQL database manager for PolyGuard.
    Same interface as ConsentDatabase from Week 8.
    FastAPI backend imports this class directly.
    """

    def __init__(self):
        self.config = DB_CONFIG

    def _conn(self):
        return psycopg2.connect(**self.config,
                                cursor_factory=RealDictCursor)

    def get_doctor(self, doctor_id):
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM doctors WHERE doctor_id=%s",
                    (doctor_id,)
                )
                return cur.fetchone()

    def authenticate_doctor(self, email, password):
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT * FROM doctors
                       WHERE email=%s AND password_hash=%s
                       AND is_active=TRUE""",
                    (email, hash_password(password))
                )
                return cur.fetchone()

    def authenticate_patient(self, phone, password):
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT * FROM patients
                       WHERE phone=%s AND password_hash=%s
                       AND is_active=TRUE""",
                    (phone, hash_password(password))
                )
                return cur.fetchone()

    def register_doctor(self, name, specialty,
                        email, phone, password):
        doctor_id = f"DR{str(uuid.uuid4())[:6].upper()}"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO doctors
                        (doctor_id, name, specialty,
                         email, phone, password_hash)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    RETURNING doctor_id
                """, (doctor_id, name, specialty,
                      email, phone, hash_password(password)))
                conn.commit()
                return doctor_id

    def register_patient(self, name, phone, age,
                         comorbidities, created_by_doctor_id,
                         date_of_birth=None, blood_group=None):
        """Doctor creates patient account — agreed project design."""
        patient_id    = f"PT{str(uuid.uuid4())[:6].upper()}"
        temp_password = f"poly{str(uuid.uuid4())[:4].upper()}"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO patients
                        (patient_id, name, phone, age,
                         date_of_birth, blood_group,
                         comorbidities, created_by_doctor,
                         password_hash)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING patient_id
                """, (patient_id, name, phone, age,
                      date_of_birth, blood_group,
                      comorbidities, created_by_doctor_id,
                      hash_password(temp_password)))
                conn.commit()
                return patient_id, temp_password

    def get_patient_by_phone(self, phone):
        """Doctor searches patient by phone number."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT * FROM patients
                       WHERE phone=%s AND is_active=TRUE""",
                    (phone,)
                )
                return cur.fetchone()

    def get_patient(self, patient_id):
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM patients WHERE patient_id=%s",
                    (patient_id,)
                )
                return cur.fetchone()

    def get_medication_profile(self, patient_id,
                               requesting_doctor_id=None):
        """Returns medications with privacy rules applied."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT m.*, d.name as doctor_name
                    FROM medications m
                    JOIN doctors d ON m.prescribed_by = d.doctor_id
                    WHERE m.patient_id=%s AND m.is_active=TRUE
                    ORDER BY m.prescribed_at
                """, (patient_id,))
                meds = cur.fetchall()

        result = []
        for med in meds:
            med = dict(med)
            privacy = med['privacy_level']

            # Patient viewing own profile
            if requesting_doctor_id is None:
                med['visible_to_doctor'] = True
                result.append(med)

            # Doctor viewing
            else:
                if privacy == 'full_share':
                    med['visible_to_doctor'] = True
                    result.append(med)
                elif privacy == 'interaction_only':
                    result.append({
                        'med_id'           : med['med_id'],
                        'drug_name'        : '*** PRIVATE ***',
                        'drug_display'     : '*** PRIVATE MEDICATION ***',
                        'dose'             : '*** HIDDEN ***',
                        'privacy_level'    : privacy,
                        'visible_to_doctor': False,
                        'interaction_check': True,
                        'patient_id'       : patient_id,
                    })
                elif privacy == 'full_private':
                    result.append({
                        'med_id'           : med['med_id'],
                        'drug_name'        : '*** PRIVATE ***',
                        'drug_display'     : '*** FULLY PRIVATE — NOT CHECKED ***',
                        'privacy_level'    : privacy,
                        'visible_to_doctor': False,
                        'interaction_check': False,
                        'patient_id'       : patient_id,
                    })
        return result

    def add_medication(self, patient_id, drug_name, dose,
                       frequency, prescribed_by,
                       privacy_level='full_share',
                       override_reason=None):
        med_id = f"MED{str(uuid.uuid4())[:5].upper()}"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO medications
                        (med_id, patient_id, drug_name,
                         drug_display, dose, frequency,
                         prescribed_by, privacy_level,
                         override_reason)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING med_id
                """, (med_id, patient_id,
                      drug_name.strip().lower(),
                      drug_name.title(), dose, frequency,
                      prescribed_by, privacy_level,
                      override_reason))
                conn.commit()
                if override_reason:
                    self.log_override(
                        patient_id, drug_name,
                        prescribed_by, 'RED',
                        0.0, override_reason
                    )
                return med_id

    def update_privacy_level(self, patient_id,
                             med_id, new_privacy_level):
        """Patient updates their own privacy flag."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                # Get current level for history
                cur.execute(
                    "SELECT privacy_level, drug_name FROM medications WHERE med_id=%s",
                    (med_id,)
                )
                existing = cur.fetchone()
                if not existing:
                    return False

                old_level = existing['privacy_level']
                drug_name = existing['drug_name']

                # Update privacy level
                cur.execute("""
                    UPDATE medications
                    SET privacy_level=%s
                    WHERE med_id=%s AND patient_id=%s
                """, (new_privacy_level, med_id, patient_id))

                # Log to consent history
                cur.execute("""
                    INSERT INTO consent_history
                        (patient_id, med_id, drug_name,
                         old_privacy, new_privacy, changed_by)
                    VALUES (%s,%s,%s,%s,%s,%s)
                """, (patient_id, med_id, drug_name,
                      old_level, new_privacy_level, patient_id))

                conn.commit()
                return True

    def log_override(self, patient_id, drug_name,
                     doctor_id, badge, risk_score, reason):
        log_id = f"OVR{str(uuid.uuid4())[:5].upper()}"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO override_logs
                        (log_id, patient_id, drug_name,
                         alert_badge, risk_score,
                         doctor_id, override_reason)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                """, (log_id, patient_id, drug_name,
                      badge, risk_score, doctor_id, reason))
                conn.commit()
                return log_id

    def get_drug_compositions(self, tablet_name):
        """Multi-ingredient tablet decomposition."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ingredient FROM drug_compositions
                    WHERE tablet_name=%s
                    ORDER BY ingredient_order
                """, (tablet_name.strip().lower(),))
                rows = cur.fetchall()
                return [r['ingredient'] for r in rows]

    def cache_interaction(self, drug1, drug2, risk_score,
                          uncertainty, badge, mechanism, side_effect):
        """Cache GNN prediction for 24-hour reuse."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO interaction_cache
                        (drug1, drug2, risk_score, uncertainty,
                         alert_badge, mechanism, side_effect)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (drug1, drug2) DO UPDATE SET
                        risk_score  = EXCLUDED.risk_score,
                        uncertainty = EXCLUDED.uncertainty,
                        alert_badge = EXCLUDED.alert_badge,
                        mechanism   = EXCLUDED.mechanism,
                        side_effect = EXCLUDED.side_effect,
                        created_at  = NOW(),
                        expires_at  = NOW() + INTERVAL '24 hours'
                """, (drug1, drug2, risk_score, uncertainty,
                      badge, mechanism, side_effect))
                conn.commit()

    def get_cached_interaction(self, drug1, drug2):
        """Return cached result if still valid."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM interaction_cache
                    WHERE drug1=%s AND drug2=%s
                    AND expires_at > NOW()
                """, (drug1, drug2))
                return cur.fetchone()

    def update_comorbidities(self, patient_id,
                             comorbidities, doctor_id):
        """Only doctors can update comorbidities."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE patients
                    SET comorbidities=%s
                    WHERE patient_id=%s
                """, (comorbidities, patient_id))
                conn.commit()
                return True

    def search_drugs(self, query, limit=10):
        """
        Autocomplete drug search — agreed project design.
        Called by /search-drugs API endpoint in Week 11.
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT tablet_name as drug_name
                    FROM drug_compositions
                    WHERE tablet_name ILIKE %s
                    UNION
                    SELECT DISTINCT ingredient as drug_name
                    FROM drug_compositions
                    WHERE ingredient ILIKE %s
                    LIMIT %s
                """, (f'%{query}%', f'%{query}%', limit))
                rows = cur.fetchall()
                return [r['drug_name'] for r in rows]

    def get_override_logs(self, patient_id=None):
        with self._conn() as conn:
            with conn.cursor() as cur:
                if patient_id:
                    cur.execute("""
                        SELECT * FROM override_logs
                        WHERE patient_id=%s
                        ORDER BY created_at DESC
                    """, (patient_id,))
                else:
                    cur.execute(
                        "SELECT * FROM override_logs ORDER BY created_at DESC"
                    )
                return cur.fetchall()


# ── STEP 6: VERIFY DATABASE ───────────────────────────────────

print("\n[STEP 6/6] Verifying database setup...")

db = PolyGuardDB()

# Table row counts
cur.execute("""
    SELECT
        (SELECT COUNT(*) FROM doctors)             as doctors,
        (SELECT COUNT(*) FROM patients)            as patients,
        (SELECT COUNT(*) FROM medications)         as medications,
        (SELECT COUNT(*) FROM drug_compositions)   as compositions,
        (SELECT COUNT(*) FROM comorbidity_profiles)as comorbidities
""")
counts = cur.fetchone()
print(f"\n  Table row counts:")
print(f"  doctors             : {counts[0]}")
print(f"  patients            : {counts[1]}")
print(f"  medications         : {counts[2]}")
print(f"  drug_compositions   : {counts[3]}")
print(f"  comorbidity_profiles: {counts[4]}")

# Test doctor auth
doc = db.authenticate_doctor('asha.n@vit.ac.in', 'doctor123')
print(f"\n  Doctor auth test    : {'PASS' if doc else 'FAIL'} — {doc['name'] if doc else 'failed'}")

# Test patient search
pt = db.get_patient_by_phone('+91-9876543210')
print(f"  Patient phone search: {'PASS' if pt else 'FAIL'} — {pt['name'] if pt else 'failed'}")

# Test medication profile
meds = db.get_medication_profile(demo_patient_id, demo_doctor_id)
print(f"  Medication profile  : {'PASS' if meds else 'FAIL'} — {len(meds)} medications loaded")

# Test drug composition
comp = db.get_drug_compositions('co-amoxiclav')
print(f"  Drug composition    : {'PASS' if comp else 'FAIL'} — co-amoxiclav → {comp}")

# Test drug search autocomplete
results = db.search_drugs('amox', limit=5)
print(f"  Drug autocomplete   : {'PASS' if results else 'FAIL'} — 'amox' → {results}")

# Save DB config for backend
db_config_export = {
    'host'             : DB_CONFIG['host'],
    'port'             : DB_CONFIG['port'],
    'database'         : DB_CONFIG['database'],
    'user'             : DB_CONFIG['user'],
    'demo_doctor_id'   : demo_doctor_id,
    'demo_patient_id'  : demo_patient_id,
    'demo_doctor_email': 'asha.n@vit.ac.in',
    'demo_doctor_pass' : 'doctor123',
    'demo_patient_phone': '+91-9876543210',
    'demo_patient_pass' : 'patient123',
}
os.makedirs('output/backend', exist_ok=True)
with open('output/backend/db_config.json', 'w') as f:
    json.dump(db_config_export, f, indent=2)

cur.close()
conn.close()

print(f"\n  output/backend/db_config.json — saved for backend use")
print(f"\n{'=' * 65}")
print(f"  Phase 3 | Week 9 COMPLETE")
print(f"  NEXT: Week 10 — FastAPI backend core endpoints")
print(f"{'=' * 65}")