import re
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Download NLTK data
nltk.download("stopwords")
nltk.download("punkt")

# Enable progress bar for Pandas
tqdm.pandas()

# Load stopwords once for efficiency
STOP_WORDS = set(stopwords.words("english"))


# Function to clean and preprocess text
def preprocess_text(text):
    """
    Clean and preprocess text data.
    """
    try:
        if pd.isna(text):  # Handle missing values
            return ""

        text = re.sub(r"http\S+|@\S+|[^a-zA-Z\s]", "", text)  # Remove URLs, mentions, and special characters
        text = text.lower()  # Convert to lowercase
        tokens = word_tokenize(text)
        filtered_text = [word for word in tokens if word not in STOP_WORDS]
        return " ".join(filtered_text)
    except Exception as e:
        logging.error(f"Error preprocessing text: {e}")
        return ""


# Function to analyze sentiment using TextBlob
def analyze_sentiment(text):
    """
    Analyze sentiment using TextBlob.
    """
    try:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return 0


# Function to classify sentiment based on polarity score
def classify_sentiment(score):
    """
    Classify sentiment as Positive, Negative, or Neutral.
    """
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"


# Load dataset with correct encoding
def load_dataset(file_path, sample_size=10000):
    """
    Load dataset from a CSV file with a specified sample size.
    """
    try:
        df = pd.read_csv(file_path, encoding="ISO-8859-1",
                         names=["polarity", "id", "date", "query", "user", "text"], header=None, nrows=sample_size)
        logging.info(f"Dataset loaded successfully with {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


# Main function
def main():
    """
    Analyze and visualize sentiment patterns in social media data.
    """
    # Use the correct local dataset path
    dataset_path = r"C:\Users\DELL\OneDrive\Desktop\PROJECTS\ProdgyInternship\sentiment140.csv"

    # Load only a sample of the data (10,000 rows for testing)
    df = load_dataset(dataset_path, sample_size=10000)

    # Ensure 'text' column exists
    if "text" not in df.columns:
        logging.error("Error: 'text' column not found in dataset!")
        print("Columns available:", df.columns)
        return

    # Preprocess and analyze sentiment with progress tracking
    df["Cleaned_Text"] = df["text"].astype(str).progress_apply(preprocess_text)
    df["Sentiment_Score"] = df["Cleaned_Text"].progress_apply(analyze_sentiment)
    df["Sentiment_Label"] = df["Sentiment_Score"].apply(classify_sentiment)

    # Analyze sentiment distribution
    sentiment_distribution = df["Sentiment_Label"].value_counts()
    logging.info(f"Sentiment Distribution:\n{sentiment_distribution}")

    # Visualize sentiment distribution using Seaborn
    plt.figure(figsize=(8, 6))
    sns.countplot(x="Sentiment_Label", data=df, palette="Set2")
    plt.title("Sentiment Distribution in Social Media Data")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

    # Interactive Pie Chart using Plotly
    fig = px.pie(sentiment_distribution, values=sentiment_distribution.values,
                 names=sentiment_distribution.index, title="Sentiment Distribution")
    fig.show()


# Run the program
if __name__ == "__main__":
    main()
