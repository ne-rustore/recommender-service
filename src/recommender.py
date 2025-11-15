import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import preprocess_data


class ContentBasedRecommender():
    def __init__(self):
        self.df = None
        self.is_trained = False
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cos_similarity = None
        self.stopwords = None
        self.ngram_size = (1, 2)

    def set_stopwords(self, stopwords):
        self.stopwords = stopwords

    def train(self, df):
        self.df = df.copy()

        self.set_stopwords(preprocess_data.get_russian_stopwords())

        self.create_tfidf_vectorizer(max_features=10000)
        self.create_tfidf_matrix()
        self.calc_similarity_matrix()

        self.is_trained = True

    def save_model(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump({'tfidf_vectorizer': self.tfidf_vectorizer,
                         'cos_similarity': self.cos_similarity,
                         'dataframe': self.df}, file)

    def load_model(self, path):
        with open(path, 'rb') as file:
            data = pickle.load(file)

        self.tfidf_vectorizer = data['tfidf_vectorizer']
        self.cos_similarity = data['cos_similarity']
        self.df = data['dataframe']
        self.is_trained = True

    def create_tfidf_vectorizer(self, max_features=10000):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=self.ngram_size, min_df=2, max_df=0.75,
            stop_words=self.stopwords, analyzer='word')

    def create_tfidf_matrix(self):
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combined_features'])

    def calc_similarity_matrix(self):
        text_similarity = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        num_features = self.df[['normalized_rating', 'normalized_rating_count']].values
        numeric_similarity = cosine_similarity(num_features, num_features)

        text_weight = 0.75
        numeric_weight = 0.25

        self.cos_similarity = (text_weight * text_similarity + numeric_weight * numeric_similarity)

        self.cos_similarity = (self.cos_similarity
                               - self.cos_similarity.min()) / (self.cos_similarity.max()
                                                               - self.cos_similarity.min())

    def get_recommendations(self, track_id, limit=5, offset=0, penalty=0.05):
        if track_id not in self.df['track_id'].values:
            return f"Приложение с ID {track_id} не найдено"

        index_track = self.df[self.df['track_id'] == track_id].index[0]

        similar_scores = list(enumerate(self.cos_similarity[index_track]))

        main_genre = self.df.iloc[index_track]['primary_genre']
        new_scores = []
        for i, score in similar_scores:
            curr_score = score
            if i != index_track and self.df.iloc[i]['primary_genre'] == main_genre:
                curr_score = score * (1 - penalty)
            new_scores.append((i, curr_score))

        similar_scores = new_scores

        similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
        similar_scores = similar_scores[offset + 1:limit + 1]

        app_indices = [i[0] for i in similar_scores]

        result_rec = self.df.iloc[app_indices][[
            'track_id', 'track_name', 'primary_genre', 'genres',
            'average_rating', 'rating_count', 'icon_url'
        ]].copy()

        sim = [i[1] for i in similar_scores]
        result_rec['similarity_score'] = sim

        if isinstance(result_rec['genres'], pd.Series):
            processed_genres = []
            for i in range(len(result_rec['genres'])):
                genres_value = result_rec['genres'].iloc[i]
                if isinstance(genres_value, str):
                    genres_list = [genre.strip() for genre in genres_value.split(',')]
                    processed_genres.append(genres_list)
                else:
                    processed_genres.append(genres_value)

            result_rec['genres'] = processed_genres

        return result_rec

    def get_trending_apps(self, limit=10, min_ratings=100, strategy_type="balanced"):
        trends = self.df[self.df['rating_count'] >= min_ratings].copy()

        if len(trends) < 1:
            return "Нет приложений с достаточным числом отзывов"

        strategies = {
            "balanced": (0.5, 0.5),
            "rating_focused": (0.9, 0.1),
            "rating_count_focused": (0.1, 0.9)
        }

        rating_weight, rating_count_weight = strategies.get(strategy_type, (0.5, 0.5))

        normalized_rating = trends['average_rating'] / 5.0
        max_rating_count = trends['rating_count'].max()
        normalized_rating_count = np.log10(trends['rating_count'] + 1) / np.log10(max_rating_count + 1)

        trend_score = (rating_weight * normalized_rating +
                       rating_count_weight * normalized_rating_count)

        trends['trend_score'] = trend_score

        trends_result = trends.nlargest(limit, 'trend_score')[[
            'track_id', 'track_name', 'primary_genre', 'genres',
            'average_rating', 'rating_count', 'trend_score', 'icon_url'
        ]].copy()

        if isinstance(trends_result['genres'], pd.Series):
            processed_genres = []
            for i in range(len(trends_result['genres'])):
                genres_value = trends_result['genres'].iloc[i]
                if isinstance(genres_value, str):
                    genres_list = [genre.strip() for genre in genres_value.split(',')]
                    processed_genres.append(genres_list)
                else:
                    processed_genres.append(genres_value)

            trends_result['genres'] = processed_genres

        return trends_result
