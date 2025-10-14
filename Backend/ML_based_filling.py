from transformers import pipeline
import re
import pandas as pd
import numpy as np
import types
import sys

# 1. Load your existing DataFrame (already has rule-based features)
df = pd.read_csv("haircare_data_enriched_features_only.csv")  # or wherever your previous CSV is

# 2. Initialize zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 3. Candidate labels
hair_patterns = ["straight", "wavy", "curly", "coily", "3a", "3b", "3c", "4a", "4b", "4c"]
hair_density = ["fine", "medium", "thick"]
scalp_conditions = ["dry scalp", "oily scalp", "flaky scalp", "sensitive scalp", "normal scalp"]
hair_concerns = ["frizz", "breakage", "dryness", "volume", "shine", "split ends", "color-treated", "heat damage"]

# 4. Clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 5. ML-based inference
def infer_features_ml(text):
    text = clean_text(text)
    if not text:
        return pd.Series([None]*4)
    
    # Hair Pattern
    pattern_res = classifier(text, hair_patterns, multi_label=True)
    if isinstance(pattern_res, types.GeneratorType):
        pattern_res = next(pattern_res)
    elif isinstance(pattern_res, list):
        pattern_res = pattern_res[0]
    labels = pattern_res.get("labels", []) if isinstance(pattern_res, dict) else []
    scores = pattern_res.get("scores", []) if isinstance(pattern_res, dict) else []
    hair_pattern = labels[0] if scores and scores[0] > 0.3 else None

    # Hair Density
    density_res = classifier(text, hair_density, multi_label=True)
    if isinstance(density_res, types.GeneratorType):
        density_res = next(density_res)
    elif isinstance(density_res, list):
        density_res = density_res[0]
    labels = density_res.get("labels", []) if isinstance(density_res, dict) else []
    scores = density_res.get("scores", []) if isinstance(density_res, dict) else []
    hair_density_label = labels[0] if scores and scores[0] > 0.3 else None

    # Scalp Condition
    scalp_res = classifier(text, scalp_conditions, multi_label=True)
    if isinstance(scalp_res, types.GeneratorType):
        scalp_res = next(scalp_res)
    elif isinstance(scalp_res, list):
        scalp_res = scalp_res[0]
    labels = scalp_res.get("labels", []) if isinstance(scalp_res, dict) else []
    scores = scalp_res.get("scores", []) if isinstance(scalp_res, dict) else []
    scalp_label = labels[0] if scores and scores[0] > 0.3 else None

    # Hair Concerns (allow multiple)
    concern_res = classifier(text, hair_concerns, multi_label=True)
    if isinstance(concern_res, types.GeneratorType):
        concern_res = next(concern_res)
    elif isinstance(concern_res, list):
        concern_res = concern_res[0]
    labels = concern_res.get("labels", []) if isinstance(concern_res, dict) else []
    scores = concern_res.get("scores", []) if isinstance(concern_res, dict) else []
    top_concerns = [c for c, s in zip(labels, scores) if s > 0.3]
    concerns_str = ", ".join(top_concerns) if top_concerns else None

    return pd.Series([hair_pattern, hair_density_label, scalp_label, concerns_str],
                     index=['HairPattern', 'HairDensity', 'ScalpCondition', 'HairConcerns'])

# Initialize columns with NaN if they don't exist
columns_to_process = ['HairPattern', 'HairDensity', 'ScalpCondition', 'HairConcerns']
for col in columns_to_process:
    if col not in df.columns:
        df[col] = np.nan

# 6. Update only missing values in-place
print("Processing reviews...")
total_reviews = len(df)

for i in range(total_reviews):
    if pd.isna(df.at[i, 'HairPattern']) or pd.isna(df.at[i, 'HairDensity']) or \
       pd.isna(df.at[i, 'ScalpCondition']) or pd.isna(df.at[i, 'HairConcerns']):
        try:
            print(f"\rProcessing review {i+1}/{total_reviews}...", end="", flush=True)
            features = infer_features_ml(df.at[i, 'Review'])
            # Update missing columns only
            for col in columns_to_process:
                if pd.isna(df.at[i, col]) and col in features.index:
                    df.at[i, col] = features[col]
        except Exception as e:
            print(f"\nError processing review {i+1}: {str(e)}", file=sys.stderr)

print("\nDone processing all reviews!")

# ✅ DataFrame is now updated in-place with ML-based inferred features
print(df.head())

# 7. Save the updated CSV (overwrite previous one)
df.to_csv("haircare_data_enriched_features_only.csv", index=False)
print("✅ ML inference applied and previous CSV updated!")
