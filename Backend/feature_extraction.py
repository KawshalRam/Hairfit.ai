import pandas as pd
import re

# 1. Load dataset
df = pd.read_csv("haircare_data.csv")

# 2. Define keyword dictionaries
hair_patterns = {
    "straight": ["straight", "smooth", "sleek"],
    "wavy": ["wavy", "waves"],
    "curly": ["curly", "curls", "ringlets"],
    "coily": ["coily", "kinky"],
    "3a": ["3a"],
    "3b": ["3b"],
    "3c": ["3c"],
    "4a": ["4a"],
    "4b": ["4b"],
    "4c": ["4c"]
}

hair_density = {
    "fine": ["fine", "thin", "light"],
    "medium": ["medium", "normal"],
    "thick": ["thick", "dense", "heavy"]
}

scalp_conditions = {
    "dry scalp": ["dry scalp", "dry skin on scalp"],
    "oily scalp": ["oily scalp", "greasy scalp"],
    "flaky scalp": ["flaky scalp", "dandruff", "flakes"],
    "sensitive scalp": ["sensitive scalp", "irritated scalp"],
    "normal scalp": ["normal scalp"]
}

hair_concerns = {
    "frizz": ["frizz", "frizzy"],
    "breakage": ["breakage", "breaks easily", "split"],
    "dryness": ["dry", "dryness"],
    "volume": ["volume", "flat", "fullness"],
    "shine": ["shine", "shiny", "glossy"],
    "split ends": ["split ends", "ends split"],
    "color-treated": ["color-treated", "dyed", "bleached"],
    "heat damage": ["heat damage", "blowdry", "flat iron", "straightener"]
}

# 3. Text cleaning
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 4. Feature extraction function
def extract_features_rule_based(text):
    text = clean_text(text)
    if not text:
        return pd.Series([None]*4)

    # Hair Pattern
    pattern = None
    for key, keywords in hair_patterns.items():
        if any(word in text for word in keywords):
            pattern = key
            break

    # Hair Density
    density = None
    for key, keywords in hair_density.items():
        if any(word in text for word in keywords):
            density = key
            break

    # Scalp Condition
    scalp = None
    for key, keywords in scalp_conditions.items():
        if any(word in text for word in keywords):
            scalp = key
            break

    # Hair Concerns
    concerns = []
    for key, keywords in hair_concerns.items():
        if any(word in text for word in keywords):
            concerns.append(key)
    concerns_str = ", ".join(concerns) if concerns else None

    return pd.Series([pattern, density, scalp, concerns_str])

# 5. Apply to dataset
df[['HairPattern', 'HairDensity', 'ScalpCondition', 'HairConcerns']] = df['Review'].apply(extract_features_rule_based)

# 6. Save enriched dataset
df.to_csv("haircare_data_enriched_features_only.csv", index=False)
print("✅ Features extracted and saved!")
