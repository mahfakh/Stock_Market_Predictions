"""
Module d'extraction des cours de bourse via yfinance.
"""

import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger("airflow.task")


def fetch_stock_data(ticker: str = "AAPL", period: str = "2y") -> pd.DataFrame:
    """
    Télécharge l'historique boursier d'un ticker via yfinance.
    Gère les en-têtes MultiIndex récents de yfinance et retourne un DataFrame nettoyé.
    """
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data is None or data.empty:
            logger.warning(f"Aucune donnée reçue pour {ticker}")
            return pd.DataFrame()

        # Gérer les colonnes MultiIndex produites par certaines versions de yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

        data = data.reset_index()

        # Vérification des colonnes minimales nécessaires
        required_cols = ["Close", "Volume"]
        for col in required_cols:
            if col not in data.columns:
                logger.error(f"Colonne requise '{col}' manquante pour {ticker}")
                return pd.DataFrame()

        return data
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement de {ticker} : {str(e)}")
        return pd.DataFrame()
