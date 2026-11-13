"""
Module d'aide à la décision et de génération des signaux de portefeuille.
"""

from typing import Tuple


def evaluate_signal(
    prediction: int,
    precision_up: float,
    precision_down: float,
    accuracy: float,
    auc: float,
) -> Tuple[str, str]:
    """
    Détermine l'action à mener (Buy, Sell, Hold) et le niveau de confiance (High, Medium, Low)
    en fonction des prédictions et des métriques d'évaluation du modèle.
    """
    if (
        precision_up > 0.60
        and (precision_up - precision_down) > 0.10
        and accuracy > 0.55
        and auc > 0.55
    ):
        action = "Buy"
        confidence = (
            "High"
            if (precision_up > 0.70 and accuracy > 0.65 and auc > 0.65)
            else "Medium"
        )
    elif (
        precision_down > 0.60
        and (precision_down - precision_up) > 0.10
        and accuracy > 0.55
        and auc > 0.55
    ):
        action = "Sell"
        confidence = (
            "High"
            if (precision_down > 0.70 and accuracy > 0.65 and auc > 0.65)
            else "Medium"
        )
    else:
        action = "Hold"
        confidence = "Low"

    return action, confidence
