"""
Tests unitaires pour les modules d'ingénierie et de prétraitement des données.
"""

import pytest
from include.config import DEFAULT_STOCK_LIST, get_storage_path
from include.evaluation import evaluate_signal

try:
    import numpy as np
    import pandas as pd
    from include.preprocess_data import FEATURE_COLUMNS, preprocess_stock_data
    PANDAS_AVAILABLE = True
except (ImportError, Exception):
    PANDAS_AVAILABLE = False


def test_config():
    """Vérifie la configuration globale."""
    assert len(DEFAULT_STOCK_LIST) > 0
    path = get_storage_path("test.csv")
    assert path.endswith("test.csv")


def test_evaluate_signal_buy():
    """Teste la règle de décision pour un signal d'achat."""
    action, confidence = evaluate_signal(
        prediction=1,
        precision_up=0.75,
        precision_down=0.40,
        accuracy=0.70,
        auc=0.72,
    )
    assert action == "Buy"
    assert confidence == "High"


def test_evaluate_signal_sell():
    """Teste la règle de décision pour un signal de vente."""
    action, confidence = evaluate_signal(
        prediction=0,
        precision_up=0.35,
        precision_down=0.72,
        accuracy=0.68,
        auc=0.69,
    )
    assert action == "Sell"
    assert confidence == "High"


def test_evaluate_signal_hold():
    """Teste la règle de décision pour un signal neutre (Hold)."""
    action, confidence = evaluate_signal(
        prediction=1,
        precision_up=0.52,
        precision_down=0.48,
        accuracy=0.51,
        auc=0.50,
    )
    assert action == "Hold"
    assert confidence == "Low"


def test_preprocess_stock_data_empty():
    """Vérifie la gestion des DataFrames vides."""
    if not PANDAS_AVAILABLE:
        pytest.skip("Pandas non disponible ou incompatible dans l'environnement hôte")
    X, y = preprocess_stock_data(pd.DataFrame())
    assert X.empty
    assert y.empty


def test_preprocess_stock_data_synthetic():
    """Teste le calcul des indicateurs sur une série de prix simulée."""
    if not PANDAS_AVAILABLE:
        pytest.skip("Pandas non disponible ou incompatible dans l'environnement hôte")
    dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(60) * 2)
    volumes = np.random.randint(1000, 50000, size=60)

    df_synth = pd.DataFrame({
        "Date": dates,
        "Close": prices,
        "Volume": volumes,
    })

    X, y = preprocess_stock_data(df_synth)

    assert not X.empty
    assert not y.empty
    assert len(X) == len(y)
    for col in FEATURE_COLUMNS:
        assert col in X.columns
    assert set(y.unique()).issubset({0, 1})
