import pandas as pd

# Load your current enriched DataFrame
df = pd.read_csv("haircare_data_enriched_features_only.csv")

# Forward fill all columns that matter
cols_to_fill = ['HairPattern', 'HairDensity', 'ScalpCondition', 'HairConcerns']

# Forward fill (propagate last valid observation forward)
df[cols_to_fill] = df[cols_to_fill].fillna(method='ffill') # type: ignore

# Optionally: Backward fill any leading NaNs (first rows)
df[cols_to_fill] = df[cols_to_fill].fillna(method='bfill') # type: ignore

# Save it back
df.to_csv("haircare_data_enriched_features_only.csv", index=False)

print("✅ Forward + backward fill complete! Dataset now dense and ready for modeling.")
