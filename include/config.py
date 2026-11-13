"""
Configuration globale du projet stock_market_predictions.
Définit la liste des actions par défaut, les chemins de stockage et les hyperparamètres.
"""

import os
import tempfile

# Répertoire de stockage des données et modèles (compatible Linux, Windows et conteneur Docker)
DEFAULT_STORAGE_DIR = os.environ.get(
    "AIRFLOW_DATA_DIR",
    os.path.join(tempfile.gettempdir(), "stock_market_predictions")
)
os.makedirs(DEFAULT_STORAGE_DIR, exist_ok=True)


def get_storage_path(filename: str) -> str:
    """Retourne le chemin absolu sécurisé pour un fichier de données ou modèle."""
    return os.path.join(DEFAULT_STORAGE_DIR, filename)


# Liste des tickers suivis par défaut (grandes capitalisations et secteurs diversifiés)
DEFAULT_STOCK_LIST = [
    "AAPL", "GOOGL", "TSLA", "MSFT", "AMZN",
    "MMM", "AOS", "ABT", "NVDA", "META"
]

# Période par défaut pour l'historique yfinance
DEFAULT_PERIOD = "2y"
