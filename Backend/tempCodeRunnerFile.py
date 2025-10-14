import pandas as pd
from transformers import pipeline
import re
import types

# 1. Load dataset
df = pd.read_csv("haircare_data.csv")

# 2. Initialize zero-shot model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 3. Define feature sets
hair_patterns = ["straight", "wavy", "curly", "coily", "3a", "3b", "3c", "4a", "4b", "4c"]
hair_density = ["fine", "medium", "thick"]
scalp_conditions = ["dry scalp", "oily scalp", "flaky scalp", "sensitive scalp", "normal scalp"]
hair_concerns = ["frizz", "breakage", "dryness", "volume", "shine", "split ends", "color-treated", "heat damage"]

# 4. Define text cleaning
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 5. Feature extraction
def extract_features(text):
    text = clean_text(text)
    if not text:
        return pd.Series([None]*5)

    # Hair Pattern
    pattern_result = classifier(text, hair_patterns, multi_label=True)
    if isinstance(pattern_result, types.GeneratorType):
        pattern_result = next(pattern_result)
    elif isinstance(pattern_result, list):
        pattern_result = pattern_result[0]
    labels = pattern_result.get("labels", []) if isinstance(pattern_result, dict) else []
    scores = pattern_result.get("scores", []) if isinstance(pattern_result, dict) else []
    hair_pattern = labels[0] if scores and scores[0] > 0.3 else None

    # Hair Density
    density_result = classifier(text, hair_density, multi_label=True)
    if isinstance(density_result, types.GeneratorType):
        density_result = next(density_result)
    elif isinstance(density_result, list):
        density_result = density_result[0]
    labels = density_result.get("labels", []) if isinstance(density_result, dict) else []
    scores = density_result.get("scores", []) if isinstance(density_result, dict) else []
    density = labels[0] if scores and scores[0] > 0.3 else None

    # Scalp Condition
    scalp_result = classifier(text, scalp_conditions, multi_label=True)
    if isinstance(scalp_result, types.GeneratorType):
        scalp_result = next(scalp_result)
    elif isinstance(scalp_result, list):
        scalp_result = scalp_result[0]
    labels = scalp_result.get("labels", []) if isinstance(scalp_result, dict) else []
    scores = scalp_result.get("scores", []) if isinstance(scalp_result, dict) else []
    scalp_condition = labels[0] if scores and scores[0] > 0.3 else None

    # Hair Concerns
    concern_result = classifier(text, hair_concerns, multi_label=True)
    if isinstance(concern_result, types.GeneratorType):
        concern_result = next(concern_result)
    elif isinstance(concern_result, list):
        concern_result = concern_result[0]
    labels = concern_result.get("labels", []) if isinstance(concern_result, dict) else []
    scores = concern_result.get("scores", []) if isinstance(concern_result, dict) else []
    top_concerns = [c for c, s in zip(labels, scores) if s > 0.3]
    concerns_str = ', '.join(top_concerns) if top_concerns else None

    # Hair Suitability Score
    score = compute_hair_suitability(text, hair_pattern, top_concerns)

    return pd.Series([hair_pattern, density, scalp_condition, concerns_str, score])

# 6. Define hair suitability computation
def compute_hair_suitability(text, pattern, concerns):
    score = 0.5  # neutral baseline

    # Positive indicators
    positives = ["love", "great", "works", "helped", "soft", "moisturized", "defined", "healthy"]
    negatives = ["not", "bad", "hate", "disappointed", "sticky", "heavy", "greasy", "dry", "brittle"]

    for word in positives:
        if word in text:
            score += 0.05
    for word in negatives:
        if word in text:
            score -= 0.05

    # Hair pattern relevance (boost if matches known pattern)
    if pattern in ["curly", "3a", "3b", "3c", "4a", "4b", "4c"]:
        if any(x in text for x in ["curl", "frizz", "moisture"]):
            score += 0.1
    elif pattern == "straight":
        if "smooth" in text or "shine" in text:
            score += 0.1

    # Concern alignment: reward if multiple concerns found
    if concerns and len(concerns) > 1:
        score += 0.05

    # Clip range between 0–1
    score = max(0, min(score, 1))
    return round(score, 2)

# 7. Apply extraction
df[['HairPattern', 'HairDensity', 'ScalpCondition', 'HairConcerns', 'HairSuitabilityScore']] = df['Review'].apply(extract_features)

# 8. Save enriched dataset
df.to_csv("haircare_data_enriched.csv", index=False)
print("✅ Enriched dataset saved with HairSuitabilityScore!")
