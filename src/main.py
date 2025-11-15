import pandas as pd
from fastapi import FastAPI
from contextlib import asynccontextmanager
import sys
import os

sys.path.append(os.path.dirname(__file__))

import preprocess_data as prep_data
from recommender import ContentBasedRecommender

recommender = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender

    try:
        model_path = 'storage/trained_model.pkl'
        data_path = 'storage/processed.csv'

        model_exists = os.path.exists(model_path)
        data_exists = os.path.exists(data_path)

        if model_exists:
            recommender = ContentBasedRecommender()
            recommender.load_model(model_path)

        else:
            if not data_exists:
                prep_data.preprocess_dataset('storage/apps.csv', data_path)

            df_processed = pd.read_csv(data_path)
            recommender = ContentBasedRecommender()
            recommender.train(df_processed)
            recommender.save_model(model_path)



    except Exception as e:
        print(f"Ошибка инициализации: {e}")

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/recommend/{track_id}")
async def get_recommendations(track_id: int, limit: int = 5, offset: int = 0):
    if recommender is None:
        return {"error": "Модель не обучена"}

    try:

        recommendations = recommender.get_recommendations(track_id, limit, offset)

        if isinstance(recommendations, str):
            return {
                "error": recommendations,
            }

        return {
            "track_id": track_id,
            "recommendations": recommendations.to_dict('records') if hasattr(recommendations,
                                                                             'to_dict') else recommendations}

    except Exception as e:
        return {"error": f"Ошибка получения рекомендаций: {e}"}


@app.get("/get_trends")
async def get_trending_apps(limit: int = 10, min_ratings: int = 100, strategy_type: str = "balanced"):
    if recommender is None:
        return {"error": "Модель не обучена"}

    try:
        best_apps = recommender.get_trending_apps(limit, min_ratings, strategy_type)
        if isinstance(best_apps, str):
            return {
                "error": best_apps,
            }

        return {
            "apps": best_apps.to_dict('records') if hasattr(best_apps, 'to_dict') else best_apps
        }

    except Exception as e:
        return {"error": f"Ошибка получения топа: {e}"}
