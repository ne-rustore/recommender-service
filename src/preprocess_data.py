import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

try:
    nltk.data.find('corpora/stopwords')
    russian_language_stopwords = stopwords.words('russian')
except LookupError:
    nltk.download('stopwords')
    russian_language_stopwords = stopwords.words('russian')


def get_russian_stopwords():
    return russian_language_stopwords


def preprocess_dataset(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path)
    stemmer = SnowballStemmer("russian")

    def preprocess_for_text(text):
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r'[^а-яёa-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        words = re.findall(r'[а-яёa-z0-9]+', text)

        processed_words = []

        for word in words:
            if len(word) > 2 and word not in russian_language_stopwords:
                if re.match(r'[а-яё]', word):
                    word = stemmer.stem(word)
                processed_words.append(word)

        return ' '.join(processed_words)

    def create_combined_features(row):
        name_weight, descript_weight, genre_weight, primary_genre_weight = 2, 3, 4, 5

        combined = (
                (row['cleaned_track_name'] + " ") * name_weight +
                (row['cleaned_description'] + " ") * descript_weight +
                (row['cleaned_genres'] + " ") * genre_weight +
                (row['cleaned_primary_genre'] + " ") * primary_genre_weight).strip()

        return combined

    df['cleaned_description'] = df['description'].apply(preprocess_for_text)
    df['cleaned_track_name'] = df['track_name'].apply(preprocess_for_text)
    df['cleaned_genres'] = df['genres'].apply(preprocess_for_text)
    df['cleaned_primary_genre'] = df['primary_genre'].apply(preprocess_for_text)

    df['combined_features'] = df.apply(create_combined_features, axis=1)

    scaler = MinMaxScaler()
    df['normalized_rating'] = scaler.fit_transform(df[['average_rating']])
    df['normalized_rating_count'] = scaler.fit_transform(df[['rating_count']])

    df.to_csv(output_file_path, index=False)
