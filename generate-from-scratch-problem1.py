# ============================================================
# BrightNest Home Synthetic Data Generator
# ============================================================
# IMPORTANT:
# Before running this code, make sure these libraries are pip installed:
# pip install pandas numpy faker
#
# This script generates synthetic CSV files for a D2C home goods company:
#   1. customers.csv
#   2. support_tickets.csv
#   3. survey_responses.csv
#   4. product_reviews.csv
#
# The generated data is designed to support an AI-powered customer insight
# prototype like the one in the BrightNest problem statement.
# ============================================================

import pandas as pd # type: ignore
import numpy  as np # type: ignore
from faker import Faker # type: ignore
from datetime import timedelta

# ============================================================
# USER-EDITABLE PARAMETERS
# ============================================================

# random seed for reproducibility
random_seed = 42

# number of customers to generate
num_customers = 500

# number of support tickets to generate
num_support_tickets = 1200

# number of survey responses to generate
num_survey_responses = 900

# number of product reviews to generate
num_product_reviews = 1000

# date range for generated events
start_date = "2025-01-01"
end_date = "2025-12-31"

# output file names
customers_file = "customers.csv"
support_tickets_file = "support_tickets.csv"
survey_responses_file = "survey_responses.csv"
product_reviews_file = "product_reviews.csv"

# approximate probabilities for data source mix
support_channel_probs = {
    "Email": 0.40,
    "Chat": 0.25,
    "Web Form": 0.20,
    "Marketplace Message": 0.10,
    "Social Media DM": 0.05
}

survey_channel_probs = {
    "Post-Purchase Survey": 0.75,
    "Email Survey": 0.25
}

review_channel_probs = {
    "Website Review": 0.70,
    "Marketplace Review": 0.30
}

# customer segments and weights
customer_segments = {
    "New Customer": 0.30,
    "Repeat Customer": 0.45,
    "Loyal Customer": 0.15,
    "At-Risk Customer": 0.10
}

# BrightNest-style product lines
product_lines = {
    "Kitchen Storage": 0.22,
    "Bathroom Essentials": 0.15,
    "Bedroom Decor": 0.18,
    "Living Room Decor": 0.20,
    "Cleaning Tools": 0.12,
    "Sustainable Accessories": 0.13
}

# issue/theme categories for synthetic feedback
themes = {
    "Shipping Delay": 0.16,
    "Damaged on Arrival": 0.10,
    "Product Quality": 0.16,
    "Wrong Item Received": 0.06,
    "Missing Parts": 0.07,
    "Packaging Waste": 0.08,
    "Website Usability": 0.07,
    "Customer Support Response": 0.10,
    "Sizing/Dimensions Mismatch": 0.06,
    "Price Concern": 0.06,
    "Loved Design": 0.04,
    "Loved Sustainability": 0.04
}

# issue severity distribution for support tickets
ticket_priorities = {
    "Low": 0.30,
    "Medium": 0.45,
    "High": 0.20,
    "Urgent": 0.05
}

# ============================================================
# INITIALIZATION
# ============================================================

fake = Faker()
np.random.seed(random_seed)
Faker.seed(random_seed)

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def weighted_choice(weight_dict):
    items = list(weight_dict.keys())
    probs = np.array(list(weight_dict.values()), dtype=float)
    probs = probs / probs.sum()
    return np.random.choice(items, p=probs)

def random_date_between(start, end):
    delta_days = (end - start).days
    rand_days = np.random.randint(0, delta_days + 1)
    return start + pd.Timedelta(days=int(rand_days))

def maybe_blank(value, prob_blank=0.15):
    return "" if np.random.rand() < prob_blank else value

def generate_customer_id(i):
    return f"CUST{i:05d}"

def generate_ticket_id(i):
    return f"TICK{i:06d}"

def generate_survey_id(i):
    return f"SURV{i:06d}"

def generate_review_id(i):
    return f"REV{i:06d}"

def choose_rating_from_theme(theme):
    # ratings aligned loosely with theme sentiment
    if theme in ["Loved Design", "Loved Sustainability"]:
        return np.random.choice([4, 5], p=[0.25, 0.75])
    elif theme in ["Price Concern", "Website Usability"]:
        return np.random.choice([2, 3, 4], p=[0.35, 0.45, 0.20])
    else:
        return np.random.choice([1, 2, 3], p=[0.35, 0.40, 0.25])

def choose_sentiment_from_theme(theme):
    if theme in ["Loved Design", "Loved Sustainability"]:
        return "Positive"
    elif theme in ["Price Concern", "Website Usability"]:
        return np.random.choice(["Negative", "Neutral"], p=[0.65, 0.35])
    else:
        return "Negative"

def survey_score_from_theme(theme):
    if theme in ["Loved Design", "Loved Sustainability"]:
        return np.random.choice([8, 9, 10], p=[0.20, 0.35, 0.45])
    elif theme in ["Price Concern", "Website Usability"]:
        return np.random.choice([4, 5, 6, 7], p=[0.20, 0.30, 0.30, 0.20])
    else:
        return np.random.choice([1, 2, 3, 4, 5, 6], p=[0.14, 0.18, 0.20, 0.20, 0.16, 0.12])

def generate_comment(theme, product_line, channel, segment):
    """
    Generates realistic text feedback tied to the chosen theme.
    This helps your later AI pipeline with tagging, summarization,
    and trend detection.
    """

    templates = {
        "Shipping Delay": [
            f"My {product_line.lower()} order arrived much later than expected.",
            f"The delivery for my {product_line.lower()} purchase took too long.",
            f"I like the product, but shipping was delayed and I was not updated.",
            f"My package for the {product_line.lower()} items came several days late."
        ],
        "Damaged on Arrival": [
            f"My {product_line.lower()} item arrived damaged.",
            f"The product was broken when I opened the package.",
            f"I received my order with visible damage on the item.",
            f"The {product_line.lower()} product looked scratched right out of the box."
        ],
        "Product Quality": [
            f"The design is nice, but the quality of the {product_line.lower()} item feels lower than expected.",
            f"The product does not feel durable enough for the price.",
            f"I expected better material quality from this {product_line.lower()} item.",
            f"The item looks good online, but in person it feels cheap."
        ],
        "Wrong Item Received": [
            f"I received the wrong item in my order.",
            f"The product delivered did not match what I purchased.",
            f"My order included a different item than the one shown online.",
            f"I got the wrong version of the product."
        ],
        "Missing Parts": [
            f"My {product_line.lower()} item was missing parts needed for assembly.",
            f"The package did not include all components.",
            f"I could not fully use the item because pieces were missing.",
            f"The product arrived incomplete."
        ],
        "Packaging Waste": [
            f"I wish the packaging used less plastic.",
            f"For a sustainable brand, the packaging felt excessive.",
            f"The item was fine, but there was too much packaging waste.",
            f"The packaging did not match the eco-friendly promise."
        ],
        "Website Usability": [
            f"It was hard to find product details on the website.",
            f"The website checkout experience was confusing.",
            f"I had trouble navigating the site while shopping for {product_line.lower()}.",
            f"The product dimensions were not easy to locate on the product page."
        ],
        "Customer Support Response": [
            f"Customer support took too long to respond.",
            f"I contacted support but did not get a helpful answer quickly.",
            f"It took multiple messages before my issue was addressed.",
            f"The support team was polite, but the response time was slow."
        ],
        "Sizing/Dimensions Mismatch": [
            f"The item dimensions were smaller than I expected from the listing.",
            f"The product did not match the size information I thought I saw online.",
            f"The storage item looked larger in the photos than in real life.",
            f"The dimensions were not clear enough before purchase."
        ],
        "Price Concern": [
            f"I like the concept, but the price feels a little high.",
            f"The product is nice, though I am not sure it is worth the price.",
            f"I expected a better value for what I paid.",
            f"The item seems overpriced compared with similar options."
        ],
        "Loved Design": [
            f"I love the look and design of this {product_line.lower()} item.",
            f"The product fits perfectly with my home aesthetic.",
            f"The design feels modern and thoughtful.",
            f"I am very happy with how this product looks in my space."
        ],
        "Loved Sustainability": [
            f"I appreciate the sustainable materials and mission behind the brand.",
            f"I bought this because I value the eco-friendly approach.",
            f"The sustainability aspect made me choose BrightNest.",
            f"I like that the product feels aligned with environmentally conscious values."
        ]
    }

    base = np.random.choice(templates[theme])

    # Add a small realistic modifier
    modifiers = [
        "",
        f" I bought it through the {channel.lower()} channel.",
        f" As a {segment.lower()}, this stood out to me.",
        " This affected my overall experience with the brand.",
        " I hope this gets improved soon.",
        " I would probably mention this in a future review."
    ]

    return base + np.random.choice(modifiers)

# ============================================================
# GENERATE CUSTOMERS TABLE
# ============================================================

customers = []

for i in range(1, num_customers + 1):
    customer_id = generate_customer_id(i)
    first_name = fake.first_name()
    last_name = fake.last_name()
    email = fake.unique.email()
    signup_date = random_date_between(start_dt - pd.Timedelta(days=730), end_dt)
    customer_segment = weighted_choice(customer_segments)
    region = np.random.choice(["Northeast", "South", "Midwest", "West"])
    preferred_channel = np.random.choice(["Email", "SMS", "Social", "Marketplace", "Website"])
    
    customers.append({
        "CustomerID": customer_id,
        "FirstName": first_name,
        "LastName": last_name,
        "Email": email,
        "SignupDate": signup_date.date(),
        "CustomerSegment": customer_segment,
        "Region": region,
        "PreferredChannel": preferred_channel
    })

customers_df = pd.DataFrame(customers)

# ============================================================
# GENERATE SUPPORT TICKETS TABLE
# ============================================================

support_tickets = []

customer_ids = customers_df["CustomerID"].tolist()

for i in range(1, num_support_tickets + 1):
    ticket_id = generate_ticket_id(i)
    customer_id = np.random.choice(customer_ids)
    event_date = random_date_between(start_dt, end_dt)
    channel = weighted_choice(support_channel_probs)
    product_line = weighted_choice(product_lines)
    theme = weighted_choice(themes)
    sentiment = choose_sentiment_from_theme(theme)
    priority = weighted_choice(ticket_priorities)

    # issue status with realistic bias
    status = np.random.choice(
        ["Resolved", "Pending", "Escalated", "Closed"],
        p=[0.55, 0.18, 0.10, 0.17]
    )

    comment_text = generate_comment(theme, product_line, channel, 
                                    customers_df.loc[customers_df["CustomerID"] == customer_id, "CustomerSegment"].values[0])

    support_tickets.append({
        "TicketID": ticket_id,
        "CustomerID": customer_id,
        "Date": event_date.date(),
        "Channel": channel,
        "ProductLine": product_line,
        "Theme": theme,
        "Sentiment": sentiment,
        "Priority": priority,
        "Status": status,
        "CommentText": comment_text
    })

support_tickets_df = pd.DataFrame(support_tickets)

# ============================================================
# GENERATE SURVEY RESPONSES TABLE
# ============================================================

survey_responses = []

for i in range(1, num_survey_responses + 1):
    survey_id = generate_survey_id(i)
    customer_id = np.random.choice(customer_ids)
    event_date = random_date_between(start_dt, end_dt)
    channel = weighted_choice(survey_channel_probs)
    product_line = weighted_choice(product_lines)
    theme = weighted_choice(themes)
    sentiment = choose_sentiment_from_theme(theme)
    satisfaction_score = survey_score_from_theme(theme)

    comment_text = generate_comment(theme, product_line, channel, 
                                    customers_df.loc[customers_df["CustomerID"] == customer_id, "CustomerSegment"].values[0])

    nps_bucket = (
        "Promoter" if satisfaction_score >= 9
        else "Passive" if satisfaction_score >= 7
        else "Detractor"
    )

    survey_responses.append({
        "SurveyResponseID": survey_id,
        "CustomerID": customer_id,
        "Date": event_date.date(),
        "Channel": channel,
        "ProductLine": product_line,
        "Theme": theme,
        "Sentiment": sentiment,
        "SatisfactionScore": satisfaction_score,
        "NPSBucket": nps_bucket,
        "CommentText": comment_text
    })

survey_responses_df = pd.DataFrame(survey_responses)

# ============================================================
# GENERATE PRODUCT REVIEWS TABLE
# ============================================================

product_reviews = []

for i in range(1, num_product_reviews + 1):
    review_id = generate_review_id(i)
    customer_id = np.random.choice(customer_ids)
    event_date = random_date_between(start_dt, end_dt)
    channel = weighted_choice(review_channel_probs)
    product_line = weighted_choice(product_lines)
    theme = weighted_choice(themes)
    sentiment = choose_sentiment_from_theme(theme)
    star_rating = choose_rating_from_theme(theme)

    review_text = generate_comment(theme, product_line, channel, 
                                   customers_df.loc[customers_df["CustomerID"] == customer_id, "CustomerSegment"].values[0])

    verified_purchase = np.random.choice(["Yes", "No"], p=[0.88, 0.12])

    product_reviews.append({
        "ReviewID": review_id,
        "CustomerID": customer_id,
        "Date": event_date.date(),
        "Channel": channel,
        "ProductLine": product_line,
        "Theme": theme,
        "Sentiment": sentiment,
        "StarRating": star_rating,
        "VerifiedPurchase": verified_purchase,
        "CommentText": review_text
    })

product_reviews_df = pd.DataFrame(product_reviews)

# ============================================================
# OPTIONAL: ADD SOME MESSINESS / REALISM
# ============================================================
# These make the data feel more realistic and useful for testing
# normalization and cleaning in your later AI pipeline.

# randomly blank out a small number of comment fields
for df in [support_tickets_df, survey_responses_df, product_reviews_df]:
    blank_mask = np.random.rand(len(df)) < 0.02
    df.loc[blank_mask, "CommentText"] = ""

# occasionally blank preferred channel
blank_pref_mask = np.random.rand(len(customers_df)) < 0.03
customers_df.loc[blank_pref_mask, "PreferredChannel"] = ""

# ============================================================
# SAVE TO CSV
# ============================================================

customers_df.to_csv(customers_file, index=False)
support_tickets_df.to_csv(support_tickets_file, index=False)
survey_responses_df.to_csv(survey_responses_file, index=False)
product_reviews_df.to_csv(product_reviews_file, index=False)

# ============================================================
# PRINT SUMMARY
# ============================================================

print("Synthetic BrightNest datasets generated successfully.")
print(f"Saved: {customers_file} ({len(customers_df)} rows)")
print(f"Saved: {support_tickets_file} ({len(support_tickets_df)} rows)")
print(f"Saved: {survey_responses_file} ({len(survey_responses_df)} rows)")
print(f"Saved: {product_reviews_file} ({len(product_reviews_df)} rows)")
