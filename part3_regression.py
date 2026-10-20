# Part 3: Features & Regression Modeling
# Project 12 - Public Transit Bus Safety

import pandas as pd
import numpy as np
from math import erfc, sqrt

df = pd.read_csv('bus_accidents_clean.csv', low_memory=False)

# -------------------------------------------------------
# Feature selection
# -------------------------------------------------------

cat_cols = ['Road_Surface_Conditions', 'Weather_Conditions',
            'Light_Conditions', 'Season', 'Day_of_Week']
num_cols = ['Speed_limit', 'Age_of_Vehicle']
target   = 'Number_of_Casualties'

# -------------------------------------------------------
# One-Hot Encoding (drop first to avoid dummy trap)
# -------------------------------------------------------

encoded = pd.get_dummies(df[cat_cols], drop_first=True)
print(f"One-Hot encoding: {len(cat_cols)} categorical columns -> {encoded.shape[1]} dummy columns")

# -------------------------------------------------------
# Min-Max Scaling for numeric columns
# -------------------------------------------------------

for col in num_cols:
    col_min = df[col].min()
    col_max = df[col].max()
    encoded[col] = (df[col] - col_min) / (col_max - col_min)
    print(f"Min-Max scaled: {col}  (min={col_min}, max={col_max})")

X = encoded.values.astype(float)
y = df[target].values.astype(float)
print(f"\nFinal feature matrix: {X.shape[0]} rows x {X.shape[1]} columns")

# -------------------------------------------------------
# PCA
# -------------------------------------------------------

X_mean = X.mean(axis=0)
X_std  = X.std(axis=0) + 1e-8
X_norm = (X - X_mean) / X_std

cov    = np.cov(X_norm.T)
eigvals, _ = np.linalg.eigh(cov)
eigvals    = eigvals[::-1]   # sort descending

explained  = eigvals / eigvals.sum()
cumulative = np.cumsum(explained)

n_keep = int(np.argmax(cumulative >= 0.90)) + 1
print(f"\nPCA:")
print(f"  Components to retain 90% variance : {n_keep}")
print(f"  Cumulative variance at {n_keep} components: {cumulative[n_keep-1]*100:.4f}%")
print(f"  Variance explained (first 5 components): {[round(float(v)*100,4) for v in explained[:5]]}%")

# -------------------------------------------------------
# Poisson Regression via IRLS (Iteratively Reweighted Least Squares)
# -------------------------------------------------------

# Add intercept
X_reg = np.column_stack([np.ones(len(X)), X])

def poisson_irls(X, y, max_iter=25, tol=1e-6):
    beta = np.zeros(X.shape[1])
    for i in range(max_iter):
        eta = X @ beta
        mu  = np.exp(np.clip(eta, -10, 10))
        z   = eta + (y - mu) / (mu + 1e-10)
        W   = mu
        XtWX = (X.T * W) @ X + np.eye(X.shape[1]) * 1e-8
        XtWz = (X.T * W) @ z
        beta_new = np.linalg.solve(XtWX, XtWz)
        if np.max(np.abs(beta_new - beta)) < tol:
            print(f"  Converged at iteration {i+1}")
            break
        beta = beta_new
    return beta

print("\nFitting Poisson Regression (IRLS)...")
beta = poisson_irls(X_reg, y)

# Standard errors from Fisher information matrix
mu_hat = np.exp(np.clip(X_reg @ beta, -10, 10))
I_mat  = (X_reg.T * mu_hat) @ X_reg
cov_b  = np.linalg.inv(I_mat + np.eye(X_reg.shape[1]) * 1e-8)
se     = np.sqrt(np.abs(np.diag(cov_b)))

z_scores = beta / (se + 1e-10)
# erfc is numerically stable for large z (avoids 1 - erf underflow)
p_vals = np.array([float(erfc(abs(z) / sqrt(2))) for z in z_scores])

# -------------------------------------------------------
# Summary table
# -------------------------------------------------------

feature_names = ['Intercept'] + list(encoded.columns)

summary = pd.DataFrame({
    'Feature'    : feature_names,
    'Coefficient': beta.round(4),
    'Std_Error'  : se.round(4),
    'Z_Score'    : z_scores.round(4),
    'p_value'    : p_vals.round(4),
    'Sig'        : ['***' if p < 0.05 else '' for p in p_vals]
})

print("\n--- Poisson Regression Summary (all features) ---")
print(f"{'Feature':<55} {'Coef':>8} {'SE':>8} {'Z':>8} {'p-value':>9} {'Sig':>5}")
print("-" * 100)
for _, row in summary.iterrows():
    print(f"  {row['Feature']:<53} {row['Coefficient']:>8.4f} {row['Std_Error']:>8.4f} "
          f"{row['Z_Score']:>8.4f} {row['p_value']:>9.4f} {row['Sig']:>5}")

sig_features = summary[summary['p_value'] < 0.05]
print(f"\nTotal features: {len(summary)} | Significant (p < 0.05): {len(sig_features)}")

# Interpret top significant features
print("\n--- Interpretation of significant variables ---")
for _, row in sig_features.sort_values('p_value').head(5).iterrows():
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    pct = round((np.exp(row['Coefficient']) - 1) * 100, 4)
    print(f"  {row['Feature']}: coef={row['Coefficient']:.4f}, p={row['p_value']:.4f} "
          f"-> casualty count {direction} by {abs(pct):.4f}% when this condition is present")

summary.to_csv('part3_poisson_summary.csv', index=False)
print("\nFull table saved: part3_poisson_summary.csv")
