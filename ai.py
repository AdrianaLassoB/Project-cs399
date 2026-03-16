# IMPORTANT:
# Before running this code, make sure these libraries are pip installed:
# pip install pandas numpy transformers torch sentence-transformers keybert

import os
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT

# ============================================================
# USER SETTINGS
# ============================================================

CUSTOMERS_FILE = "customers.csv"
SUPPORT_FILE = "support_tickets.csv"
SURVEY_FILE = "survey_responses.csv"
REVIEWS_FILE = "product_reviews.csv"

SAVE_TEXT_SUMMARY = True
TEXT_SUMMARY_FILE = "brightnest_summary.txt"

TOP_N_EXAMPLES = 3
MAX_TEXT_CHARS = 500
BATCH_SIZE = 32

INCLUDE_OVERALL_SUMMARY = True

# Optional natural-language search query
# Example: "customers complaining about damaged items"
SEARCH_QUERY = "customers complaining about product quality and damaged items"
TOP_SEARCH_RESULTS = 5

THEME_DEFINITIONS = {
    "Shipping Delay": "The customer says the order arrived late, shipping was delayed, or delivery took too long.",
    "Damaged on Arrival": "The customer says the item arrived broken, scratched, cracked, or damaged.",
    "Product Quality": "The customer says the product feels cheap, low quality, or not durable.",
    "Wrong Item Received": "The customer says they received the wrong product or wrong version.",
    "Missing Parts": "The customer says the order or item was incomplete or missing components.",
    "Packaging Waste": "The customer complains about too much packaging, plastic, or waste.",
    "Website Usability": "The customer had trouble using the website, product page, or checkout.",
    "Customer Support Response": "The customer says support was slow, unhelpful, or took too long to respond.",
    "Sizing/Dimensions Mismatch": "The customer says the size or dimensions were unclear or different than expected.",
    "Price Concern": "The customer says the item is too expensive, overpriced, or not worth the price.",
    "Loved Design": "The customer praises the design, style, or appearance of the product.",
    "Loved Sustainability": "The customer praises the eco-friendly or sustainable aspects of the brand or product."
}

# ============================================================
# VALIDATION HELPERS
# ============================================================

def ensure_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

def load_csv_safe(file_path, required_columns):
    ensure_file_exists(file_path)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(f"Could not read {file_path}: {e}")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{file_path} is missing required columns: {missing_cols}")

    return df

def safe_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip()

def add_header(lines, title, char="="):
    lines.append(title)
    lines.append(char * len(title))

# ============================================================
# LOAD SOURCE FILES
# ============================================================

try:
    customers_df = load_csv_safe(CUSTOMERS_FILE, ["CustomerID"])
    support_df = load_csv_safe(
        SUPPORT_FILE,
        ["CustomerID", "Date", "Channel", "ProductLine", "CommentText"]
    )
    survey_df = load_csv_safe(
        SURVEY_FILE,
        ["CustomerID", "Date", "Channel", "ProductLine", "CommentText"]
    )
    reviews_df = load_csv_safe(
        REVIEWS_FILE,
        ["CustomerID", "Date", "Channel", "ProductLine", "CommentText"]
    )
except Exception as e:
    print(f"\nERROR: {e}")
    print("Please fix the input files and run the script again.")
    raise SystemExit(1)

# ============================================================
# STANDARDIZE AND COMBINE FEEDBACK
# ============================================================

support_std = support_df[["CustomerID", "Date", "Channel", "ProductLine", "CommentText"]].copy()
support_std["Source"] = "Support Ticket"

survey_std = survey_df[["CustomerID", "Date", "Channel", "ProductLine", "CommentText"]].copy()
survey_std["Source"] = "Survey Response"

reviews_std = reviews_df[["CustomerID", "Date", "Channel", "ProductLine", "CommentText"]].copy()
reviews_std["Source"] = "Product Review"

feedback_df = pd.concat([support_std, survey_std, reviews_std], ignore_index=True)

feedback_df["CommentText"] = feedback_df["CommentText"].apply(safe_text)
feedback_df["Date"] = pd.to_datetime(feedback_df["Date"], errors="coerce")
feedback_df["Channel"] = feedback_df["Channel"].fillna("Unknown").astype(str).str.strip()
feedback_df["ProductLine"] = feedback_df["ProductLine"].fillna("Unknown").astype(str).str.strip()
feedback_df["Source"] = feedback_df["Source"].fillna("Unknown").astype(str).str.strip()

feedback_df = feedback_df[feedback_df["CommentText"] != ""].copy()
feedback_df = feedback_df[feedback_df["Date"].notna()].copy()

if feedback_df.empty:
    print("ERROR: No usable customer feedback rows were found after cleaning.")
    raise SystemExit(1)

feedback_df["Month"] = feedback_df["Date"].dt.to_period("M").astype(str)

# ============================================================
# LOAD AI MODELS
# ============================================================

print("Loading AI models...")

try:
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
except Exception as e:
    print(f"\nERROR loading sentiment model: {e}")
    raise SystemExit(1)

try:
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
except Exception as e:
    print(f"\nERROR loading sentence-transformer model: {e}")
    raise SystemExit(1)

try:
    keyword_model = KeyBERT(model=embedding_model)
except Exception as e:
    print(f"\nERROR loading KeyBERT: {e}")
    raise SystemExit(1)

theme_names = list(THEME_DEFINITIONS.keys())
theme_texts = list(THEME_DEFINITIONS.values())
theme_embeddings = embedding_model.encode(theme_texts, convert_to_tensor=True)

# ============================================================
# AI FUNCTIONS
# ============================================================

def predict_sentiment_batch(texts):
    cleaned = []
    for text in texts:
        t = safe_text(text)[:MAX_TEXT_CHARS]
        cleaned.append(t if t else "No comment provided.")

    try:
        results = sentiment_model(cleaned, batch_size=BATCH_SIZE)
    except Exception:
        return ["Unknown"] * len(cleaned)

    labels = []
    for result in results:
        label = str(result["label"]).upper()
        if "POS" in label:
            labels.append("Positive")
        elif "NEG" in label:
            labels.append("Negative")
        else:
            labels.append(label.title())
    return labels

def predict_theme_batch(texts):
    cleaned = []
    for text in texts:
        t = safe_text(text)[:MAX_TEXT_CHARS]
        cleaned.append(t if t else "No comment provided.")

    try:
        text_embeddings = embedding_model.encode(cleaned, convert_to_tensor=True)
        similarities = util.cos_sim(text_embeddings, theme_embeddings)
    except Exception:
        return ["Unknown"] * len(cleaned)

    predicted = []
    for i in range(len(cleaned)):
        best_idx = int(similarities[i].argmax())
        predicted.append(theme_names[best_idx])

    return predicted

def extract_keywords_for_group(texts, top_n=5):
    joined_text = " ".join([safe_text(t) for t in texts if safe_text(t) != ""]).strip()

    if not joined_text:
        return []

    try:
        keywords = keyword_model.extract_keywords(
            joined_text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n
        )
        return [kw for kw, score in keywords]
    except Exception:
        return []

def semantic_search(query, comments, top_n=5):
    clean_comments = [safe_text(c)[:MAX_TEXT_CHARS] for c in comments]
    clean_comments = [c for c in clean_comments if c != ""]

    if not query.strip() or not clean_comments:
        return []

    try:
        query_embedding = embedding_model.encode(query, convert_to_tensor=True)
        comment_embeddings = embedding_model.encode(clean_comments, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, comment_embeddings)[0]

        top_indices = similarities.argsort(descending=True)[:top_n]
        results = []
        for idx in top_indices:
            idx_int = int(idx)
            results.append(clean_comments[idx_int])
        return results
    except Exception:
        return []

# ============================================================
# RUN AI
# ============================================================

print("Analyzing customer feedback with AI...")

all_comments = feedback_df["CommentText"].tolist()
feedback_df["PredictedSentiment"] = predict_sentiment_batch(all_comments)
feedback_df["PredictedTheme"] = predict_theme_batch(all_comments)

# ============================================================
# SUMMARY CALCULATIONS
# ============================================================

sentiment_counts = feedback_df["PredictedSentiment"].value_counts()

top_themes_overall = (
    feedback_df["PredictedTheme"]
    .value_counts()
    .reset_index()
)
top_themes_overall.columns = ["Theme", "Count"]

negative_df = feedback_df[feedback_df["PredictedSentiment"] == "Negative"].copy()

top_negative_themes = (
    negative_df["PredictedTheme"]
    .value_counts()
    .reset_index()
)
top_negative_themes.columns = ["Theme", "Count"]

monthly_negative = (
    negative_df.groupby(["Month", "PredictedTheme"])
    .size()
    .reset_index(name="Count")
    .sort_values(["Month", "Count"], ascending=[True, False])
)

channel_negative = (
    negative_df.groupby(["Channel", "PredictedTheme"])
    .size()
    .reset_index(name="Count")
    .sort_values(["Channel", "Count"], ascending=[True, False])
)

review_only_df = feedback_df[feedback_df["Source"] == "Product Review"].copy()

negative_review_counts_by_product = (
    review_only_df[review_only_df["PredictedSentiment"] == "Negative"]
    .groupby("ProductLine")
    .size()
    .sort_values(ascending=False)
)

# ============================================================
# BUILD TEXT SUMMARY
# ============================================================

lines = []

add_header(lines, "BRIGHTNEST HOME - AI CUSTOMER INSIGHT SUMMARY")
lines.append("")
lines.append(f"Total customer records: {len(customers_df)}")
lines.append(f"Total feedback records analyzed: {len(feedback_df)}")
lines.append(f"Support tickets loaded: {len(support_df)}")
lines.append(f"Survey responses loaded: {len(survey_df)}")
lines.append(f"Product reviews loaded: {len(reviews_df)}")
lines.append("")

if INCLUDE_OVERALL_SUMMARY:
    add_header(lines, "OVERALL CROSS-CHANNEL SUMMARY", "-")
    lines.append("")
    lines.append("Sentiment breakdown:")
    for label, count in sentiment_counts.items():
        lines.append(f"  - {label}: {count}")
    lines.append("")

    lines.append("Top themes overall:")
    for _, row in top_themes_overall.head(5).iterrows():
        lines.append(f"  - {row['Theme']}: {row['Count']}")
    lines.append("")

    lines.append("Top negative complaints overall:")
    for _, row in top_negative_themes.head(5).iterrows():
        lines.append(f"  - {row['Theme']}: {row['Count']}")
    lines.append("")

add_header(lines, "PRODUCT REVIEW SUMMARY BY PRODUCT LINE", "-")
lines.append("")

if review_only_df.empty:
    lines.append("No product review data was available.")
else:
    if negative_review_counts_by_product.empty:
        lines.append("No negative product reviews were identified.")
        lines.append("")
    else:
        lines.append("Product lines with the most negative reviews:")
        for product, count in negative_review_counts_by_product.head(5).items():
            lines.append(f"  - {product}: {count}")
        lines.append("")

    for product in sorted(review_only_df["ProductLine"].dropna().unique().tolist()):
        product_df = review_only_df[review_only_df["ProductLine"] == product].copy()
        negative_product_df = product_df[product_df["PredictedSentiment"] == "Negative"].copy()
        positive_product_df = product_df[product_df["PredictedSentiment"] == "Positive"].copy()

        sentiment_counts_product = product_df["PredictedSentiment"].value_counts()

        top_negative_product_themes = (
            negative_product_df["PredictedTheme"]
            .value_counts()
            .head(3)
        )

        top_positive_product_themes = (
            positive_product_df["PredictedTheme"]
            .value_counts()
            .head(3)
        )

        product_keywords = extract_keywords_for_group(product_df["CommentText"].tolist(), top_n=5)

        lines.append(product)
        lines.append("~" * len(product))
        lines.append(f"Total product reviews analyzed: {len(product_df)}")

        for sentiment_label in ["Negative", "Positive", "Neutral", "Unknown"]:
            if sentiment_label in sentiment_counts_product:
                lines.append(f"  {sentiment_label} reviews: {sentiment_counts_product[sentiment_label]}")

        lines.append("  Top complaints:")
        if len(top_negative_product_themes) == 0:
            lines.append("    - None identified.")
        else:
            for theme, count in top_negative_product_themes.items():
                lines.append(f"    - {theme}: {count}")

        lines.append("  Top praise:")
        if len(top_positive_product_themes) == 0:
            lines.append("    - None identified.")
        else:
            for theme, count in top_positive_product_themes.items():
                lines.append(f"    - {theme}: {count}")

        lines.append("  Top keywords/phrases from reviews:")
        if not product_keywords:
            lines.append("    - None identified.")
        else:
            for kw in product_keywords:
                lines.append(f"    - {kw}")

        lines.append("  Example negative review comments:")
        neg_examples = negative_product_df["CommentText"].head(TOP_N_EXAMPLES).tolist()
        if not neg_examples:
            lines.append("    - None available.")
        else:
            for i, comment in enumerate(neg_examples, start=1):
                lines.append(f"    {i}. {comment}")

        lines.append("  Example positive review comments:")
        pos_examples = positive_product_df["CommentText"].head(TOP_N_EXAMPLES).tolist()
        if not pos_examples:
            lines.append("    - None available.")
        else:
            for i, comment in enumerate(pos_examples, start=1):
                lines.append(f"    {i}. {comment}")

        lines.append("")

add_header(lines, "SEMANTIC SEARCH RESULTS", "-")
lines.append(f"Query: {SEARCH_QUERY}")
search_results = semantic_search(SEARCH_QUERY, feedback_df["CommentText"].tolist(), top_n=TOP_SEARCH_RESULTS)

if not search_results:
    lines.append("No matching comments found.")
else:
    for i, result in enumerate(search_results, start=1):
        lines.append(f"  {i}. {result}")
lines.append("")

add_header(lines, "MONTHLY NEGATIVE TRENDS (TOP 10)", "-")
if monthly_negative.empty:
    lines.append("No negative monthly trends found.")
else:
    for _, row in monthly_negative.head(10).iterrows():
        lines.append(f"  - {row['Month']} | {row['PredictedTheme']} | {row['Count']}")
lines.append("")

add_header(lines, "NEGATIVE THEMES BY CHANNEL (TOP 10)", "-")
if channel_negative.empty:
    lines.append("No negative channel patterns found.")
else:
    for _, row in channel_negative.head(10).iterrows():
        lines.append(f"  - {row['Channel']} | {row['PredictedTheme']} | {row['Count']}")
lines.append("")

summary_text = "\n".join(lines)

# ============================================================
# OUTPUT
# ============================================================

print("\n" + summary_text)

if SAVE_TEXT_SUMMARY:
    try:
        with open(TEXT_SUMMARY_FILE, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"\nSaved text summary: {TEXT_SUMMARY_FILE}")
    except Exception as e:
        print(f"\nCould not save text summary file: {e}")