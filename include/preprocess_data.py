"""
Module de calcul des indicateurs techniques et de préparation des données pour les modèles ML.
"""

from typing import Tuple
import numpy as np
import pandas as pd
import ta

FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "SMA_5",
    "SMA_20",
    "RSI",
    "MACD",
    "Daily_Return",
    "Volatility",
    "Momentum",
]


def preprocess_stock_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Calcule les indicateurs techniques et génère la variable cible (Target = 1 si hausse à J+1).

    Retourne:
        X (pd.DataFrame): DataFrame contenant les features calculées.
        y (pd.Series): Série contenant la variable cible binaire.
    """
    if df is None or df.empty or len(df) < 30:
        return pd.DataFrame(), pd.Series(dtype=int)

    df = df.copy()

    # Nettoyage et conversion numérique
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # Indicateurs de tendance et de dynamique
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd_diff(df["Close"])

    # Rendement journalier, volatilité et momentum
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Close"].rolling(window=5).std() / df["Close"]
    df["Momentum"] = df["Close"] - df["Close"].rolling(window=10).mean()

    # Variable cible : 1 si le cours de clôture de demain est supérieur à aujourd'hui, 0 sinon
    df["Tomorrow_Close"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow_Close"] > df["Close"]).astype(int)

    # Suppression des lignes avec valeurs manquantes (générées par les fenêtres glissantes et shift)
    df = df.dropna().reset_index(drop=True)

    if df.empty or len(df) < 10:
        return pd.DataFrame(), pd.Series(dtype=int)

    X = df[FEATURE_COLUMNS].copy()
    y = df["Target"].copy()

    return X, y
