# =============================================================================
# Project 12: Public Transit Bus Safety Under High-Friction Asset Decay
# Part 1: Ingestion & Data Cleaning
# Dataset: UK Department for Transport Road Safety Open Data
# Source:  https://github.com/Vishwesh4/UK-road-safety-kaggle
# =============================================================================

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# STEP 1: Load the two relational tables
# -----------------------------------------------------------------------------
print("=" * 65)
print("STEP 1: Loading datasets")
print("=" * 65)

acc = pd.read_csv('Accident_Information.csv', encoding='latin-1', low_memory=False)
print(f"  Accident_Information.csv loaded  ->  {len(acc):,} rows, {len(acc.columns)} columns")

veh = pd.read_csv('Vehicle_Information.csv', encoding='latin-1', low_memory=False)
print(f"  Vehicle_Information.csv loaded   ->  {len(veh):,} rows, {len(veh.columns)} columns")

# -----------------------------------------------------------------------------
# STEP 2: Filter for buses and merge tables on Accident_Index
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 2: Filtering buses and linking tables")
print("=" * 65)

bus_veh = veh[veh['Vehicle_Type'] == 'Bus or coach (17 or more pass seats)'].copy()
print(f"  Bus rows after Vehicle_Type filter  ->  {len(bus_veh):,} rows")

df = pd.merge(bus_veh, acc, on='Accident_Index', how='inner')
print(f"  After inner join on Accident_Index  ->  {len(df):,} rows, {len(df.columns)} columns")

# -----------------------------------------------------------------------------
# STEP 3: Missing data audit — percentage of NaN per column
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 3: Missing Data Audit (columns with at least 1 null entry)")
print("=" * 65)

missing = pd.DataFrame({
    'Column'        : df.columns,
    'Missing_Count' : df.isnull().sum().values,
    'Missing_Pct'   : (df.isnull().sum().values / len(df) * 100).round(4)
})
missing = missing[missing['Missing_Count'] > 0].sort_values('Missing_Pct', ascending=False)

print(f"\n{'Column':<50} {'Missing Count':>15} {'Missing %':>12}")
print("-" * 80)
for _, row in missing.iterrows():
    print(f"  {row['Column']:<48} {int(row['Missing_Count']):>15,} {row['Missing_Pct']:>11.4f}%")
print("-" * 80)
print(f"  Total columns: {len(df.columns)}   |   Columns with nulls: {len(missing)}")

# -----------------------------------------------------------------------------
# STEP 4: Fill categorical NaN — null means "event did not occur"
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 4: Categorical NaN fill (null = event did not occur -> 'None')")
print("=" * 65)

event_cols = [
    'Carriageway_Hazards',
    'Hit_Object_in_Carriageway',
    'Hit_Object_off_Carriageway',
    'Special_Conditions_at_Site',
    'Skidding_and_Overturning'
]
for col in event_cols:
    before = df[col].isnull().sum()
    df[col] = df[col].fillna('None')
    print(f"  {col:<42}  filled {before:,} nulls -> 'None'")

df['model'] = df['model'].fillna('Unknown')
df['make']  = df['make'].fillna('Unknown')
print(f"  {'model':<42}  filled with 'Unknown'")
print(f"  {'make':<42}  filled with 'Unknown'")

# -----------------------------------------------------------------------------
# STEP 5: Validity rules — flag out-of-range values, replace with median
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 5: Validity Rules & Median Replacement")
print("=" * 65)

validity_results = []

# Rule 1: Speed_limit — UK legal range is 0 to 70 mph
mask1 = (df['Speed_limit'] < 0) | (df['Speed_limit'] > 70)
med1  = df.loc[~mask1, 'Speed_limit'].median()
df.loc[mask1, 'Speed_limit'] = med1
validity_results.append({
    'Column'   : 'Speed_limit',
    'Rule'     : 'Speed_limit < 0 or > 70 mph',
    'Flagged'  : int(mask1.sum()),
    'Median'   : round(med1, 4)
})

# Rule 2: Age_of_Vehicle — buses older than 50 years are registry errors
mask2 = df['Age_of_Vehicle'] > 50
med2  = df.loc[~mask2, 'Age_of_Vehicle'].median()
df.loc[mask2, 'Age_of_Vehicle'] = med2
validity_results.append({
    'Column'   : 'Age_of_Vehicle',
    'Rule'     : 'Age_of_Vehicle > 50 years',
    'Flagged'  : int(mask2.sum()),
    'Median'   : round(med2, 4)
})

# Rule 3: Engine_Capacity_.CC. — bus engines outside 1,000–50,000 CC are erroneous
mask3 = (df['Engine_Capacity_.CC.'] < 1000) | (df['Engine_Capacity_.CC.'] > 50000)
med3  = df.loc[~mask3, 'Engine_Capacity_.CC.'].median()
df.loc[mask3, 'Engine_Capacity_.CC.'] = med3
validity_results.append({
    'Column'   : 'Engine_Capacity_.CC.',
    'Rule'     : 'Engine_Capacity < 1,000 or > 50,000 CC',
    'Flagged'  : int(mask3.sum()),
    'Median'   : round(med3, 4)
})

print(f"\n{'Column':<25} {'Rule':<45} {'Flagged':>8} {'Median':>10}")
print("-" * 95)
for r in validity_results:
    print(f"  {r['Column']:<23} {r['Rule']:<45} {r['Flagged']:>8,} {r['Median']:>10.4f}")
print("-" * 95)

# -----------------------------------------------------------------------------
# STEP 6: Fill remaining numeric NaN columns with column medians
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 6: Numeric NaN fill (column median)")
print("=" * 65)

numeric_fill_cols = ['Age_of_Vehicle', 'Engine_Capacity_.CC.', 'Driver_IMD_Decile']
for col in numeric_fill_cols:
    n_null = df[col].isnull().sum()
    if n_null > 0:
        med = df[col].median()
        df[col] = df[col].fillna(med)
        print(f"  {col:<35}  filled {n_null:,} nulls with median = {med:.4f}")
    else:
        print(f"  {col:<35}  no nulls remaining")

# -----------------------------------------------------------------------------
# STEP 7: Parse Date, extract Month, Year, Season
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 7: Date parsing — extract Month, Year, Season")
print("=" * 65)

df['Date']    = pd.to_datetime(df['Date'], errors='coerce')
df['Month']   = df['Date'].dt.month
df['Year_num'] = df['Date'].dt.year
df['Season']  = df['Month'].map({
    12:'Winter', 1:'Winter',  2:'Winter',
    3:'Spring',  4:'Spring',  5:'Spring',
    6:'Summer',  7:'Summer',  8:'Summer',
    9:'Autumn',  10:'Autumn', 11:'Autumn'
})
print(f"  Season distribution:\n{df['Season'].value_counts().to_string()}")

# -----------------------------------------------------------------------------
# STEP 8: Save cleaned dataset
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 8: Saving cleaned dataset")
print("=" * 65)

df.to_csv('bus_accidents_clean.csv', index=False)
print(f"  Saved: bus_accidents_clean.csv  ->  {len(df):,} rows, {len(df.columns)} columns")
print("\nPart 1 complete.")
