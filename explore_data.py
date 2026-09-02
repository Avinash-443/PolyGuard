import pandas as pd

# DrugBank DDI
df_ddi = pd.read_csv('Dataset/db_drug_interactions.csv')
print("=== DrugBank DDI ===")
print("Shape:", df_ddi.shape)
print("Columns:", df_ddi.columns.tolist())
print(df_ddi.head(3))

# TWOSIDES
df_two = pd.read_csv('Dataset/TWOSIDES.csv.gz', compression='gzip')
print("\n=== TWOSIDES ===")
print("Shape:", df_two.shape)
print("Columns:", df_two.columns.tolist())
print(df_two.head(3))

# OFFSIDES
df_off = pd.read_csv('Dataset/OFFSIDES.csv.gz', compression='gzip')
print("\n=== OFFSIDES ===")
print("Shape:", df_off.shape)
print("Columns:", df_off.columns.tolist())
print(df_off.head(3))

# Effect similarities
df_sim = pd.read_csv('Dataset/offsides_effect_similarities.csv.gz', compression='gzip')
print("\n=== Effect Similarities ===")
print("Shape:", df_sim.shape)
print("Columns:", df_sim.columns.tolist())
print(df_sim.head(3))