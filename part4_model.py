# Part 4: AI Failure Audit
# Random Forest to predict accident severity (Safe / Minor / Severe)

import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.read_csv('bus_accidents_clean.csv', low_memory=False)

# -------------------------------------------------------
# Labels
# -------------------------------------------------------

label_map  = {'Slight': 0, 'Serious': 1, 'Fatal': 2}
label_name = {0: 'Safe', 1: 'Minor', 2: 'Severe'}
df['label'] = df['Accident_Severity'].map(label_map)

# -------------------------------------------------------
# Features
# -------------------------------------------------------

df['is_wet']    = df['Road_Surface_Conditions'].str.contains('Wet',      na=False).astype(int)
df['is_frost']  = df['Road_Surface_Conditions'].str.contains('Frost|ice',na=False).astype(int)
df['is_night']  = df['Light_Conditions'].str.contains('Darkness',        na=False).astype(int)
df['is_summer'] = (df['Season'] == 'Summer').astype(int)

feat_cols = ['Speed_limit', 'Age_of_Vehicle', 'Number_of_Vehicles',
             'is_wet', 'is_frost', 'is_night', 'is_summer']

X = df[feat_cols].values.astype(float)
y = df['label'].values.astype(int)

# -------------------------------------------------------
# Train / test split (80 / 20)
# -------------------------------------------------------

idx   = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

print("Class distribution in full dataset:")
for c, name in label_name.items():
    print(f"  {name}: {(y == c).sum()} ({(y == c).mean()*100:.1f}%)")

# Balance training set so the model learns all three classes equally
safe_idx   = np.where(y_train == 0)[0]
minor_idx  = np.where(y_train == 1)[0]
severe_idx = np.where(y_train == 2)[0]
n_each = len(severe_idx)   # match smallest class

balanced = np.concatenate([
    np.random.choice(safe_idx,  n_each, replace=False),
    np.random.choice(minor_idx, n_each, replace=False),
    severe_idx
])
X_bal = X_train[balanced]
y_bal = y_train[balanced]
print(f"\nBalanced training size: {len(X_bal)} rows ({n_each} per class)")

# -------------------------------------------------------
# Simple Decision Tree (depth = 3)
# -------------------------------------------------------

def gini(y):
    n = len(y)
    if n == 0: return 0
    return 1 - sum((np.sum(y == c) / n) ** 2 for c in [0, 1, 2])

def best_split(X, y):
    best_gain, best_col, best_thresh = -1, 0, 0
    base = gini(y)
    for col in range(X.shape[1]):
        for thresh in np.unique(np.percentile(X[:, col], [25, 50, 75])):
            left  = y[X[:, col] <= thresh]
            right = y[X[:, col] >  thresh]
            if len(left) == 0 or len(right) == 0:
                continue
            gain = base - (len(left)/len(y) * gini(left) + len(right)/len(y) * gini(right))
            if gain > best_gain:
                best_gain, best_col, best_thresh = gain, col, thresh
    return best_col, best_thresh

def build_tree(X, y, depth):
    if depth == 0 or len(np.unique(y)) == 1 or len(y) < 5:
        return {'label': int(np.bincount(y, minlength=3).argmax())}
    col, thresh = best_split(X, y)
    left_mask   = X[:, col] <= thresh
    return {
        'col': col, 'thresh': thresh,
        'left':  build_tree(X[left_mask],  y[left_mask],  depth - 1),
        'right': build_tree(X[~left_mask], y[~left_mask], depth - 1)
    }

def predict(node, X):
    if 'label' in node:
        return np.full(len(X), node['label'], dtype=int)
    mask = X[:, node['col']] <= node['thresh']
    preds = np.empty(len(X), dtype=int)
    if mask.any():   preds[mask]  = predict(node['left'],  X[mask])
    if (~mask).any(): preds[~mask] = predict(node['right'], X[~mask])
    return preds

# -------------------------------------------------------
# Random Forest: 10 trees, majority vote
# -------------------------------------------------------

print("\nTraining Random Forest (10 trees, depth=3)...")
trees = []
for i in range(10):
    boot = np.random.choice(len(X_bal), len(X_bal), replace=True)
    trees.append(build_tree(X_bal[boot], y_bal[boot], depth=3))
    print(f"  Tree {i+1}/10 trained")

all_preds = np.array([predict(t, X_test) for t in trees])
y_pred    = np.array([np.bincount(all_preds[:, i], minlength=3).argmax()
                      for i in range(len(X_test))])

# -------------------------------------------------------
# Precision, Recall, F1
# -------------------------------------------------------

print(f"\n{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 45)
f1_list = []
for c in [0, 1, 2]:
    tp = ((y_pred == c) & (y_test == c)).sum()
    fp = ((y_pred == c) & (y_test != c)).sum()
    fn = ((y_pred != c) & (y_test == c)).sum()
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    f1_list.append(f1)
    print(f"  {label_name[c]:<8} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")

macro_f1 = sum(f1_list) / 3
print(f"  {'Macro':<8} {'':>10} {'':>10} {macro_f1:>10.4f}")

# -------------------------------------------------------
# False Negatives: predicted Safe but actually Severe
# -------------------------------------------------------

fn_mask = (y_pred == 0) & (y_test == 2)
fn_df   = df.iloc[idx[split:]][fn_mask][
    ['Speed_limit', 'Road_Surface_Conditions', 'Light_Conditions',
     'Number_of_Vehicles', 'Weather_Conditions', 'Accident_Severity']
].copy()

print(f"\nFalse Negatives (predicted Safe, actual Severe): {fn_mask.sum()} rows")
print(fn_df.head(10).to_string(index=False))

print("\nFalse Negative Profile:")
print("  Road surface:  ", fn_df['Road_Surface_Conditions'].value_counts().to_dict())
print("  Light:         ", fn_df['Light_Conditions'].value_counts().to_dict())
print("  Speed limit:   ", fn_df['Speed_limit'].value_counts().head(4).to_dict())

fn_df.to_csv('part4_false_negatives.csv', index=False)
print("\nSaved: part4_false_negatives.csv")
