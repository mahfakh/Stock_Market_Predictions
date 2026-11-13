"""
DAG Apache Airflow : Pipeline complet de prédiction boursière avec XGBoost.
Étapes :
1. Extraction des cours boursiers (yfinance)
2. Prétraitement et calcul des indicateurs techniques (SMA, RSI, MACD, etc.)
3. Entraînement et optimisation d'hyperparamètres (XGBoost + TimeSeriesSplit)
4. Prédiction de la tendance à J+1 sur la dernière séance
5. Génération des signaux et recommandations de portefeuille (Buy / Sell / Hold)
"""

from datetime import datetime, timedelta
import logging
import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from airflow import DAG
from airflow.operators.python import PythonOperator

from include.config import DEFAULT_STOCK_LIST, get_storage_path
from include.evaluation import evaluate_signal
from include.fetch_stock_data import fetch_stock_data
from include.preprocess_data import FEATURE_COLUMNS, preprocess_stock_data

logger = logging.getLogger("airflow.task")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def fetch_data_task(**kwargs):
    """Télécharge l'historique de chaque action et enregistre le CSV brut."""
    for ticker in DEFAULT_STOCK_LIST:
        try:
            df = fetch_stock_data(ticker=ticker, period="2y")
            if not df.empty:
                raw_path = get_storage_path(f"{ticker}_raw.csv")
                df.to_csv(raw_path, index=False)
                logger.info(f"[{ticker}] Données brutes sauvegardées : {raw_path}")
            else:
                logger.warning(f"[{ticker}] Données vides, ticker ignoré.")
        except Exception as e:
            logger.error(f"[{ticker}] Erreur lors de l'extraction : {str(e)}")


def preprocess_data_task(**kwargs):
    """Calcule les indicateurs techniques et la variable cible."""
    for ticker in DEFAULT_STOCK_LIST:
        raw_path = get_storage_path(f"{ticker}_raw.csv")
        if not os.path.exists(raw_path):
            logger.warning(f"[{ticker}] Fichier brut non trouvé, prétraitement ignoré.")
            continue

        try:
            df_raw = pd.read_csv(raw_path)
            X, y = preprocess_stock_data(df_raw)

            if not X.empty and not y.empty:
                df_prep = X.copy()
                df_prep["Target"] = y.values
                prep_path = get_storage_path(f"{ticker}_preprocessed.csv")
                df_prep.to_csv(prep_path, index=False)
                logger.info(f"[{ticker}] Données prétraitées sauvegardées : {prep_path}")
            else:
                logger.warning(f"[{ticker}] Données insuffisantes après prétraitement.")
        except Exception as e:
            logger.error(f"[{ticker}] Erreur prétraitement : {str(e)}")


def train_model_task(**kwargs):
    """Entraîne un modèle XGBoost avec TimeSeriesSplit et GridSearchCV."""
    for ticker in DEFAULT_STOCK_LIST:
        prep_path = get_storage_path(f"{ticker}_preprocessed.csv")
        if not os.path.exists(prep_path):
            continue

        try:
            df = pd.read_csv(prep_path)
            X = df[FEATURE_COLUMNS]
            y = df["Target"]

            if len(X) < 40:
                logger.warning(f"[{ticker}] Trop peu d'échantillons ({len(X)}) pour l'entraînement.")
                continue

            tscv = TimeSeriesSplit(n_splits=5)
            best_model = None
            best_f1 = -1.0
            last_test_idx = None

            param_grid = {
                "n_estimators": [50, 100],
                "max_depth": [3, 5],
                "learning_rate": [0.03, 0.1],
                "subsample": [0.8, 1.0],
            }

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                grid = GridSearchCV(
                    XGBClassifier(
                        random_state=42,
                        eval_metric="logloss",
                        use_label_encoder=False if hasattr(XGBClassifier, "use_label_encoder") else None,
                    ),
                    param_grid,
                    cv=3,
                    scoring="f1",
                    n_jobs=1,
                )
                grid.fit(X_train, y_train)
                candidate = grid.best_estimator_

                preds = candidate.predict(X_test)
                score = f1_score(y_test, preds, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_model = candidate
                    last_test_idx = test_idx

            if best_model is not None and last_test_idx is not None:
                X_eval = X.iloc[last_test_idx]
                y_eval = y.iloc[last_test_idx]
                preds = best_model.predict(X_eval)

                report = classification_report(y_eval, preds, output_dict=True, zero_division=0)
                precision_up = float(report.get("1", {}).get("precision", 0.0))
                precision_down = float(report.get("0", {}).get("precision", 0.0))

                try:
                    proba = best_model.predict_proba(X_eval)[:, 1]
                    auc = float(roc_auc_score(y_eval, proba))
                except Exception:
                    auc = float(accuracy_score(y_eval, preds))

                acc = float(accuracy_score(y_eval, preds))

                model_path = get_storage_path(f"{ticker}_xgboost_model.pkl")
                joblib.dump(best_model, model_path)

                metrics_df = pd.DataFrame([{
                    "Ticker": ticker,
                    "Model": "XGBoost",
                    "Precision_Up": round(precision_up, 4),
                    "Precision_Down": round(precision_down, 4),
                    "F1_Score": round(best_f1, 4),
                    "ROC_AUC": round(auc, 4),
                    "Accuracy": round(acc, 4),
                }])
                metrics_path = get_storage_path(f"{ticker}_xgboost_metrics.csv")
                metrics_df.to_csv(metrics_path, index=False)

                logger.info(
                    f"[{ticker}] Modèle entraîné. Acc={acc:.3f}, AUC={auc:.3f}, "
                    f"P_Up={precision_up:.3f}, P_Down={precision_down:.3f}"
                )
        except Exception as e:
            logger.error(f"[{ticker}] Erreur entraînement : {str(e)}")


def predict_trend_task(**kwargs):
    """Effectue la prédiction de la tendance pour la prochaine séance."""
    for ticker in DEFAULT_STOCK_LIST:
        model_path = get_storage_path(f"{ticker}_xgboost_model.pkl")
        prep_path = get_storage_path(f"{ticker}_preprocessed.csv")
        metrics_path = get_storage_path(f"{ticker}_xgboost_metrics.csv")

        if not (os.path.exists(model_path) and os.path.exists(prep_path) and os.path.exists(metrics_path)):
            continue

        try:
            model = joblib.load(model_path)
            df = pd.read_csv(prep_path)
            metrics_df = pd.read_csv(metrics_path)

            latest_features = df.iloc[-1:][FEATURE_COLUMNS]
            prediction = int(model.predict(latest_features)[0])

            precision_up = float(metrics_df["Precision_Up"].iloc[0])
            precision_down = float(metrics_df["Precision_Down"].iloc[0])

            result_df = pd.DataFrame([{
                "Ticker": ticker,
                "Model": "XGBoost",
                "Prediction": "Up" if prediction == 1 else "Down",
                "Prediction_Numeric": prediction,
                "Precision_Up": precision_up,
                "Precision_Down": precision_down,
            }])
            pred_path = get_storage_path(f"{ticker}_xgboost_prediction.csv")
            result_df.to_csv(pred_path, index=False)
            logger.info(f"[{ticker}] Prédiction enregistrée : {'Up' if prediction == 1 else 'Down'}")
        except Exception as e:
            logger.error(f"[{ticker}] Erreur prédiction : {str(e)}")


def recommend_portfolio_task(**kwargs):
    """Agrège les prédictions et métriques pour émettre les signaux de portefeuille."""
    signals = []
    today_str = datetime.now().strftime("%Y-%m-%d")

    for ticker in DEFAULT_STOCK_LIST:
        pred_path = get_storage_path(f"{ticker}_xgboost_prediction.csv")
        metrics_path = get_storage_path(f"{ticker}_xgboost_metrics.csv")

        if not (os.path.exists(pred_path) and os.path.exists(metrics_path)):
            continue

        try:
            pred_df = pd.read_csv(pred_path)
            metrics_df = pd.read_csv(metrics_path)

            pred_num = int(pred_df["Prediction_Numeric"].iloc[0])
            prediction_label = str(pred_df["Prediction"].iloc[0])
            precision_up = float(metrics_df["Precision_Up"].iloc[0])
            precision_down = float(metrics_df["Precision_Down"].iloc[0])
            accuracy = float(metrics_df["Accuracy"].iloc[0])
            auc = float(metrics_df["ROC_AUC"].iloc[0])

            action, confidence = evaluate_signal(
                prediction=pred_num,
                precision_up=precision_up,
                precision_down=precision_down,
                accuracy=accuracy,
                auc=auc,
            )

            signals.append({
                "Date": today_str,
                "Ticker": ticker,
                "Model": "XGBoost",
                "Prediction": prediction_label,
                "Action": action,
                "Confidence": confidence,
                "Precision_Up": precision_up,
                "Precision_Down": precision_down,
                "Accuracy": accuracy,
                "ROC_AUC": auc,
            })
        except Exception as e:
            logger.error(f"[{ticker}] Erreur recommandation : {str(e)}")

    if signals:
        out_df = pd.DataFrame(signals)
        out_path = get_storage_path(f"stock_signals_xgboost_{today_str}.csv")
        out_df.to_csv(out_path, index=False)
        logger.info(f"Recommandations de portefeuille générées : {out_path}")
        print(out_df.to_string(index=False))


with DAG(
    dag_id="stock_prediction_xgboost_pipeline",
    default_args=default_args,
    description="Pipeline de prédiction de tendance boursière utilisant XGBoost",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["stock", "mlops", "xgboost", "production"],
) as dag:

    fetch_task = PythonOperator(
        task_id="fetch_stock_data",
        python_callable=fetch_data_task,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_stock_data",
        python_callable=preprocess_data_task,
    )

    train_task = PythonOperator(
        task_id="train_xgboost_models",
        python_callable=train_model_task,
    )

    predict_task = PythonOperator(
        task_id="predict_stock_trends",
        python_callable=predict_trend_task,
    )

    recommend_task = PythonOperator(
        task_id="recommend_portfolio",
        python_callable=recommend_portfolio_task,
    )

    fetch_task >> preprocess_task >> train_task >> predict_task >> recommend_task
