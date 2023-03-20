from datetime import datetime
from typing import List, Dict

import matplotlib as plt
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from JobSearcher import JobSearcher


# %%

def analyze_job_ads(job_ads: List[Dict[str, str]]) -> pd.DataFrame:
    # Extract dates and documents from the list of dictionaries
    dates = [format_date(ad["Publication_Date"], "day") for ad in job_ads]
    documents = [ad["Job Title"] for ad in job_ads]

    # Create a BERTopic model with a pre-trained sentence transformer
    sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    topic_model = BERTopic(embedding_model=sentence_model, min_topic_size=1)

    # Fit and transform the model on the documents
    topics, _ = topic_model.fit_transform(documents)

    # Create a DataFrame with the results
    results = pd.DataFrame({"date": dates, "job_ad": documents, "topic": topics})

    # Convert dates to datetime objects and sort the DataFrame by date
    results["date"] = pd.to_datetime(results["date"])
    results = results.sort_values("date")

    # Group the results by date and topic, then count the occurrences
    topic_counts = results.groupby(["date", "topic"]).size().reset_index(name="count")

    return topic_counts


def display_changes_over_time(topic_counts: pd.DataFrame):
    # Pivot the DataFrame to show the changes in topics over time
    pivot_table = topic_counts.pivot(index="date", columns="topic", values="count").fillna(0)

    # Plot the changes in topics over time
    ax = pivot_table.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_ylabel("Number of job ads")
    ax.set_title("Changes in Job Ad Topics Over Time")
    ax.legend(title="Topic")

    plt.show()


def format_date(date_str: str, format_option: str) -> str:
    # Parse the date string
    date_obj = datetime.fromisoformat(date_str)

    # Format the date according to the given option
    if format_option == "day":
        formatted_date = date_obj.strftime("%Y-%m-%d")
    elif format_option == "month":
        formatted_date = date_obj.strftime("%Y-%m")
    elif format_option == "year":
        formatted_date = date_obj.strftime("%Y")
    else:
        raise ValueError("Invalid format option. Choose from 'day', 'month', or 'year'.")

    return formatted_date


# %% Search for jobs

titles = ["developer", "programmer", "data scientist", "data engineer", "data analyst", "machine learning engineer",
          "machine learning scientist", "machine learning analyst", "software engineer", "software developer"]

js = JobSearcher(titles=titles, page=100, search_limit=100)

result = js.search_jobs()
df = pd.DataFrame(result)

# %% Analyze job ads

topic_counts = analyze_job_ads(result)

# %% Display changes over time

display_changes_over_time(topic_counts)
