# hybrid_recommender.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import math


def build_combined_features(df,
                            cols=['HairPattern', 'HairDensity', 'ScalpCondition', 'HairConcerns', 'Brand', 'Product']):
    """
    Combine multiple descriptive features into a single text column for TF-IDF vectorization.
    """
    def safe_join(row):
        pieces = []
        for c in cols:
            v = row.get(c, '')
            if pd.isna(v):
                v = ''
            if isinstance(v, list):
                v = ' '.join(v)
            pieces.append(str(v))
        return ' '.join(pieces)

    df['combined_features'] = df.apply(safe_join, axis=1)
    return df


def build_content_model(df, text_col='combined_features', max_features=5000, ngram_range=(1, 2)):
    """
    TF-IDF vectorization for the combined features.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(df[text_col].fillna(''))
    return vectorizer, tfidf_matrix


def compute_signals(df,
                    suitability_col='HairSuitabilityScore',
                    rating_col='Rating',
                    n_ratings_col='RatingNumeric'):
    """
    Normalize and combine numeric signals for each product (rating, popularity, suitability, etc.).
    """
    signals = pd.DataFrame(index=df.index)

    # Suitability (optional)
    if suitability_col in df.columns:
        signals['suitability'] = pd.to_numeric(df[suitability_col], errors='coerce')
    else:
        signals['suitability'] = np.nan

    # Ratings
    if rating_col in df.columns:
        signals['rating'] = pd.to_numeric(df[rating_col], errors='coerce')
    else:
        signals['rating'] = np.nan

    # Popularity proxy — number of ratings
    if n_ratings_col in df.columns:
        signals['pop'] = pd.to_numeric(df[n_ratings_col], errors='coerce').fillna(0).apply(lambda x: math.log(x + 1))
    else:
        signals['pop'] = np.nan

    # Fill NaNs with median or fallback
    signals['suitability'] = signals['suitability'].fillna(
        signals['suitability'].median() if not signals['suitability'].isna().all() else 0.5
    )
    signals['rating'] = signals['rating'].fillna(
        signals['rating'].median() if not signals['rating'].isna().all() else 4.0
    )
    signals['pop'] = signals['pop'].fillna(
        signals['pop'].median() if not signals['pop'].isna().all() else 0.0
    )

    # Normalize to [0,1]
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(signals[['suitability', 'rating', 'pop']])
    signals['norm_suit'], signals['norm_rating'], signals['norm_pop'] = norm[:, 0], norm[:, 1], norm[:, 2]

    return signals


def create_recommender(df, vectorizer=None, tfidf_matrix=None, signals_df=None):
    """
    Create a hybrid recommender using content + signal weighting + optional concern boosts.
    """
    if vectorizer is None or tfidf_matrix is None:
        df = build_combined_features(df)
        vectorizer, tfidf_matrix = build_content_model(df)

    if signals_df is None:
        signals_df = compute_signals(df)

    # fix column names
    idx_to_product = df.reset_index()[['index', 'Product', 'Brand']]

    def recommend(user_profile, top_n=10, alpha=0.6, beta=0.3, gamma=0.1, concerns_boost=None):
        """
        Recommend top products based on hybrid scoring.
        """
        # 1️⃣ Build user text
        parts = []
        for k in ['HairPattern', 'HairDensity', 'ScalpCondition', 'HairConcerns', 'Brand', 'Product']:
            v = user_profile.get(k, '')
            if v is None:
                v = ''
            if isinstance(v, list):
                v = ' '.join(v)
            parts.append(str(v))
        user_text = ' '.join(parts)

        # 2️⃣ Compute cosine similarity
        user_vec = vectorizer.transform([user_text])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()

        # 3️⃣ Combine numeric signals
        norm_suit = np.asarray(signals_df['norm_suit'].values, dtype=np.float64)
        norm_rating = np.asarray(signals_df['norm_rating'].values, dtype=np.float64)
        norm_pop = np.asarray(signals_df['norm_pop'].values, dtype=np.float64)

        signal_score = 0.5 * norm_suit + 0.4 * norm_rating + 0.1 * norm_pop
        if signal_score.max() > signal_score.min():
            signal_score = (signal_score - signal_score.min()) / (signal_score.max() - signal_score.min())
        else:
            signal_score = np.zeros_like(signal_score)

        # 4️⃣ Optional concern boost
        concern_bonus = np.zeros_like(signal_score)
        if concerns_boost and user_profile.get('HairConcerns'):
            user_concerns = [c.strip().lower() for c in str(user_profile.get('HairConcerns')).split(',')]
            for i, row_text in enumerate(df['combined_features'].fillna('').str.lower()):
                bonus = 0.0
                for uc in user_concerns:
                    if uc and uc in row_text:
                        bonus += concerns_boost.get(uc, 0.05)
                concern_bonus[i] = min(bonus, 0.5)

        # 5️⃣ Hybrid scoring
        denom = alpha + beta + gamma if (alpha + beta + gamma) != 0 else 1.0
        final_score = (alpha * sims + beta * signal_score + gamma * concern_bonus) / denom

        # 6️⃣ Aggregate & sort
        res = df.copy().reset_index()
        res['cosine_sim'] = sims
        res['signal_score'] = signal_score
        res['concern_bonus'] = concern_bonus
        res['final_score'] = final_score

        recs = res.sort_values('final_score', ascending=False).head(top_n)
        cols = ['index', 'Product', 'Brand', 'cosine_sim', 'signal_score', 'concern_bonus', 'final_score']
        return recs[cols]

    return recommend, vectorizer, tfidf_matrix, signals_df


if __name__ == "__main__":
    # 🔹 Load enriched dataset
    df = pd.read_csv("haircare_data_enriched_features_only.csv")
    df = build_combined_features(df)
    vectorizer, tfidf_matrix = build_content_model(df)
    signals_df = compute_signals(df)

    # 🔹 Initialize recommender
    recommend_fn, _, _, _ = create_recommender(df, vectorizer, tfidf_matrix, signals_df)

    # 🔹 Example user profile
    user_profile = {
        "HairPattern": "curly",
        "HairDensity": "thick",
        "ScalpCondition": "dry scalp",
        "HairConcerns": "frizz, dryness"
    }

    # 🔹 Run recommendations
    recs = recommend_fn(
        user_profile,
        top_n=10,
        alpha=0.6,
        beta=0.35,
        gamma=0.05,
        concerns_boost={'frizz': 0.08, 'dryness': 0.06}
    )

    print(recs.to_string(index=False))
