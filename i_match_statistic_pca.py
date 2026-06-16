import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('misc', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Load the data
print("Loading data...")
df = pd.read_csv('matchhistory_data/match_statistics.csv')

# Store game_id and goal columns if they exist
game_ids = None
home_goals = None
away_goals = None

if 'game_id' in df.columns:
    game_ids = df['game_id'].copy()
    print("Found game_id column")

# Check for goal columns (FTHG/HG and FTAG/AG)
if 'FTHG' in df.columns:
    home_goals = df['FTHG'].copy()
    print("Found FTHG column (Full Time Home Goals)")
elif 'HG' in df.columns:
    home_goals = df['HG'].copy()
    print("Found HG column (Home Goals)")

if 'FTAG' in df.columns:
    away_goals = df['FTAG'].copy()
    print("Found FTAG column (Full Time Away Goals)")
elif 'AG' in df.columns:
    away_goals = df['AG'].copy()
    print("Found AG column (Away Goals)")

# Define the features for PCA
features = [
    'HS', 'AS', 'HST', 'AST', 'HHW', 'AHW', 
    'HC', 'AC', 'HF', 'AF', 'HFKC', 'AFKC',
    'HO', 'AO', 'HY', 'AY', 'HR', 'AR'
]

# Check which features are actually present in the dataset
available_features = [f for f in features if f in df.columns]
missing_features = [f for f in features if f not in df.columns]

if missing_features:
    print(f"Warning: The following features are missing from the dataset: {missing_features}")
    print(f"Proceeding with available features: {available_features}")

# Create a subset with only the features for PCA
X = df[available_features].copy()

# Check for missing values
print(f"\nMissing values before processing:")
print(X.isnull().sum())

# Create mask for rows that have all features available
valid_rows_mask = X.notna().all(axis=1)

# Handle missing values - drop rows with any missing values
X_clean = X.dropna()
print(f"\nRows before dropping NA: {len(X)}")
print(f"Rows after dropping NA: {len(X_clean)}")

# Filter game_ids and goals to match cleaned data
if game_ids is not None:
    game_ids_clean = game_ids[valid_rows_mask].reset_index(drop=True)
    X_clean = X_clean.reset_index(drop=True)

if home_goals is not None:
    home_goals_clean = home_goals[valid_rows_mask].reset_index(drop=True)

if away_goals is not None:
    away_goals_clean = away_goals[valid_rows_mask].reset_index(drop=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# Perform PCA
print("\nPerforming PCA...")
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Create DataFrame with PCA results, game_id, and goals
pca_results = pd.DataFrame(
    X_pca,
    columns=[f'PC{i+1}' for i in range(len(available_features))]
)

# Add game_id if available
if game_ids is not None:
    pca_results.insert(0, 'game_id', game_ids_clean)

# Add goal columns if available
if home_goals is not None:
    pca_results['HG'] = home_goals_clean
    print("Added HG column to PCA results")

if away_goals is not None:
    pca_results['AG'] = away_goals_clean
    print("Added AG column to PCA results")
    
# Save PCA results with game_id and goals to data/ subfolder
pca_results.to_csv('data/match_statistics_pca.csv', index=False)
print("\nPCA results saved to 'data/match_statistics_pca.csv'")
print(f"Dataset shape: {pca_results.shape}")
print(f"Columns: {list(pca_results.columns)}")

# --- SAVE ALL OTHER FILES TO misc/ SUBFOLDER ---

# Save the scaler parameters
scaler_params = pd.DataFrame({
    'feature': available_features,
    'mean': scaler.mean_,
    'scale': scaler.scale_
})
scaler_params.to_csv('misc/scaler_params.csv', index=False)
print("Scaler parameters saved to 'misc/scaler_params.csv'")

# Create loadings DataFrame
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(available_features))],
    index=available_features
)

# Save loadings to CSV
loadings.to_csv('misc/pca_loadings.csv')
print("Loadings saved to 'misc/pca_loadings.csv'")

# Save explained variance information
variance_df = pd.DataFrame({
    'PC': [f'PC{i+1}' for i in range(len(available_features))],
    'Explained_Variance_Ratio': explained_variance_ratio,
    'Cumulative_Variance_Ratio': cumulative_variance_ratio,
    'Eigenvalue': explained_variance_ratio * len(available_features)
})
variance_df.to_csv('misc/explained_variance.csv', index=False)
print("Explained variance saved to 'misc/explained_variance.csv'")

# Create detailed Excel report
with pd.ExcelWriter('misc/pca_results.xlsx') as writer:
    # Sheet 1: Loadings
    loadings.to_excel(writer, sheet_name='Loadings')
    
    # Sheet 2: Explained variance
    variance_df.to_excel(writer, sheet_name='Explained_Variance', index=False)
    
    # Sheet 3-7: Top features for first 5 PCs
    for i in range(min(5, len(available_features))):
        pc_col = f'PC{i+1}'
        top_features = loadings[pc_col].abs().sort_values(ascending=False).head(10)
        top_features_df = pd.DataFrame({
            'Feature': top_features.index,
            'Loading': loadings.loc[top_features.index, pc_col].values,
            'Absolute_Loading': top_features.values
        })
        top_features_df.to_excel(writer, sheet_name=f'Top_Features_PC{i+1}', index=False)
    
    # Sheet: Scaler parameters
    scaler_params.to_excel(writer, sheet_name='Scaler_Params', index=False)

print("Detailed results saved to 'misc/pca_results.xlsx'")

# Save PCA model for future use (using joblib)
try:
    import joblib
    joblib.dump(pca, 'misc/pca_model.joblib')
    joblib.dump(scaler, 'misc/scaler.joblib')
    print("PCA model and scaler saved to 'misc/' folder")
except ImportError:
    print("Note: joblib not installed. PCA model not saved. Install with: pip install joblib")

# Print summary statistics
print("\n" + "="*60)
print("PCA ANALYSIS SUMMARY")
print("="*60)
print(f"\nNumber of features used: {len(available_features)}")
print(f"Number of samples: {len(X_clean)}")

print("\nExplained Variance by Principal Components:")
for i, (var, cum_var) in enumerate(zip(explained_variance_ratio[:10], cumulative_variance_ratio[:10])):
    print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%) - Cumulative: {cum_var:.3f} ({cum_var*100:.1f}%)")

# Find number of components needed for common thresholds
thresholds = [0.70, 0.80, 0.90, 0.95]
print(f"\nComponents needed for variance thresholds:")
for threshold in thresholds:
    n_comp = np.argmax(cumulative_variance_ratio >= threshold) + 1
    reduction = (1 - n_comp/len(available_features)) * 100
    print(f"  {threshold*100:.0f}%: {n_comp} components ({reduction:.1f}% reduction)")

# Kaiser criterion
eigenvalues = explained_variance_ratio * len(available_features)
kaiser_n = sum(eigenvalues > 1)
print(f"\nKaiser criterion (eigenvalue > 1): {kaiser_n} components")

# Print top 5 features for first 3 PCs
print("\nTop 5 features for first 3 PCs:")
for i in range(min(3, len(available_features))):
    pc_col = f'PC{i+1}'
    print(f"\n{pc_col} ({explained_variance_ratio[i]*100:.1f}% variance):")
    top_features = loadings[pc_col].abs().sort_values(ascending=False).head(5)
    for feat, loading in zip(top_features.index, loadings.loc[top_features.index, pc_col]):
        direction = "+" if loading > 0 else "-"
        print(f"  {feat}: {loading:.3f} ({direction})")

# Create visualization plots and save to plots/ subfolder
print("\nCreating visualizations...")

# Set up style
plt.style.use('default')
sns.set_palette("husl")

# 1. Scree plot with thresholds
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Principal Component', fontsize=12)
ax.set_ylabel('Explained Variance Ratio', fontsize=12)
ax.set_title('Scree Plot - Explained Variance by Principal Component', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add annotations for variance explained
for i, var in enumerate(explained_variance_ratio[:5]):
    ax.annotate(f'{var*100:.1f}%', 
                (i+1, var), 
                textcoords="offset points",
                xytext=(0,10), 
                ha='center',
                fontsize=9)

plt.tight_layout()
plt.savefig('plots/scree_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Cumulative variance plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'ro-', linewidth=2, markersize=8)
ax.axhline(y=0.70, color='orange', linestyle='--', linewidth=1.5, label='70% threshold')
ax.axhline(y=0.80, color='g', linestyle='--', linewidth=2, label='80% threshold')
ax.axhline(y=0.90, color='b', linestyle='--', linewidth=2, label='90% threshold')
ax.axhline(y=0.95, color='purple', linestyle='--', linewidth=1.5, label='95% threshold')
ax.set_xlabel('Number of Principal Components', fontsize=12)
ax.set_ylabel('Cumulative Explained Variance Ratio', fontsize=12)
ax.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig('plots/cumulative_variance_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. First two PCs scatter plot (if at least 2 PCs available)
if len(available_features) >= 2:
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, c=X_pca[:, 0], cmap='viridis', edgecolors='black', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)', fontsize=12)
    ax.set_title('First Two Principal Components', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='grey', linestyle='-', linewidth=0.5)
    plt.colorbar(scatter, label='PC1 Score')
    plt.tight_layout()
    plt.savefig('plots/pc1_vs_pc2_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Loading plots for first 4 PCs
n_pcs_to_plot = min(4, len(available_features))
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i in range(n_pcs_to_plot):
    pc_col = f'PC{i+1}'
    loadings_sorted = loadings[pc_col].sort_values()
    
    colors = ['#d73027' if x < 0 else '#4575b4' for x in loadings_sorted.values]
    axes[i].barh(range(len(loadings_sorted)), loadings_sorted.values, color=colors, alpha=0.8)
    axes[i].set_yticks(range(len(loadings_sorted)))
    axes[i].set_yticklabels(loadings_sorted.index, fontsize=9)
    axes[i].set_xlabel('Loading Value', fontsize=10)
    axes[i].set_title(f'PC{i+1} Loadings ({explained_variance_ratio[i]:.1%} variance)', fontsize=12, fontweight='bold')
    axes[i].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[i].grid(True, alpha=0.3, axis='x')

# Hide unused subplots if less than 4 PCs
for i in range(n_pcs_to_plot, 4):
    axes[i].set_visible(False)

plt.suptitle('PCA Loadings by Principal Component', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/loadings_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Heatmap of loadings
fig, ax = plt.subplots(figsize=(12, len(available_features) * 0.5))
n_pcs_heatmap = min(8, len(available_features))
im = ax.imshow(loadings.iloc[:, :n_pcs_heatmap].values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(n_pcs_heatmap))
ax.set_xticklabels([f'PC{i+1}\n({explained_variance_ratio[i]:.1%})' for i in range(n_pcs_heatmap)], fontsize=9)
ax.set_yticks(range(len(available_features)))
ax.set_yticklabels(available_features, fontsize=9)
ax.set_title('PCA Loadings Heatmap', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Loading Value')
plt.tight_layout()
plt.savefig('plots/loadings_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Explained variance pie chart for first 5 PCs (if available)
if len(available_features) >= 5:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    sizes = list(explained_variance_ratio[:5])
    sizes.append(1 - sum(sizes))  # Remaining variance
    labels = [f'PC{i+1}' for i in range(5)] + ['Others']
    colors = plt.cm.Set3(range(6))
    explode = [0.05] * 5 + [0]
    
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode)
    ax1.set_title('Variance Explained by Top 5 PCs', fontsize=14, fontweight='bold')
    
    # Bar chart comparison
    components = [f'PC{i+1}' for i in range(5)]
    bars = ax2.bar(components, explained_variance_ratio[:5], color=colors[:5], alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Principal Components', fontsize=12)
    ax2.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax2.set_title('Variance Explained - Top 5 PCs', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, var in zip(bars, explained_variance_ratio[:5]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{var*100:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('plots/variance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

print("All visualizations saved to 'plots/' subfolder")
print("\n" + "="*60)
print("PCA ANALYSIS COMPLETE")
print("="*60)
print(f"\nFile summary:")
print(f"  data/match_statistics_pca.csv - PCA transformed data with game_id, HG, AG")
print(f"  misc/pca_loadings.csv - PCA loadings matrix")
print(f"  misc/explained_variance.csv - Explained variance statistics")
print(f"  misc/scaler_params.csv - Standardization parameters")
print(f"  misc/pca_results.xlsx - Detailed Excel report")
print(f"  misc/pca_model.joblib - PCA model (if joblib installed)")
print(f"  plots/ - All visualization plots")