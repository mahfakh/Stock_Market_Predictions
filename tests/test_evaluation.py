"""
Tests unitaires pour la configuration et le moteur de décision (Buy, Sell, Hold).
"""

from include.config import DEFAULT_STOCK_LIST, get_storage_path
from include.evaluation import evaluate_signal


def test_config_paths():
    assert len(DEFAULT_STOCK_LIST) >= 5
    path = get_storage_path("AAPL_model.pkl")
    assert path.endswith("AAPL_model.pkl")


def test_evaluate_signal_buy():
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
    action, confidence = evaluate_signal(
        prediction=1,
        precision_up=0.52,
        precision_down=0.48,
        accuracy=0.51,
        auc=0.50,
    )
    assert action == "Hold"
    assert confidence == "Low"
