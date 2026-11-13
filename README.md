# 📈 Stock Market Predictions - Pipeline MLOps avec Apache Airflow & Astronomer

Pipeline MLOps automatisé de bout en bout pour la **prédiction directionnelle des cours boursiers** et la **génération de signaux d'aide à la décision de portefeuille** (*Buy / Sell / Hold*), orchestré avec **Apache Airflow** (via **Astronomer Astro CLI**).

---

## 📑 Sommaire
- [1. Présentation du Projet](#1-présentation-du-projet)
- [2. Architecture & Flux de Données](#2-architecture--flux-de-données)
- [3. Structure du Répertoire](#3-structure-du-répertoire)
- [4. Pipelines de Machine Learning (DAGs)](#4-pipelines-de-machine-learning-dags)
- [5. Feature Engineering & Indicateurs Techniques](#5-feature-engineering--indicateurs-techniques)
- [6. Moteur de Décision & Signaux de Portefeuille](#6-moteur-de-décision--signaux-de-portefeuille)
- [7. Bonnes Pratiques MLOps Appliquées](#7-bonnes-pratiques-mlops-appliquées)
- [8. Installation et Guide de Démarrage](#8-installation-et-guide-de-démarrage)
- [9. Tests & Validation](#9-tests--validation)
- [10. Roadmap & Évolutions Futures](#10-roadmap--évolutions-futures)

---

## 1. Présentation du Projet

Ce projet a pour objectif de prédire quotidiennement si le cours de clôture d'une action à $J+1$ sera en hausse par rapport à sa clôture de $J$ (classification binaire) pour un panier d'actions majeures (ex: *AAPL, MSFT, GOOGL, TSLA, AMZN, NVDA, META, MMM, AOS, ABT*).

À partir de ces prédictions et d'une analyse rigoureuse des métriques de performance des modèles (Précision à la hausse, Précision à la baisse, ROC-AUC, Accuracy), le pipeline émet chaque jour une recommandation d'arbitrage :
- **🟢 Buy** : Forte probabilité et précision historique sur la hausse.
- **🔴 Sell** : Forte probabilité et précision historique sur la baisse.
- **🟡 Hold** : Incertitude ou divergence de performance, maintien de position.

---

## 2. Architecture & Flux de Données

Le pipeline s'exécute selon un graphe orienté acyclique (DAG) quotidien en 5 étapes séquentielles :

```
┌─────────────────────────┐
│   fetch_stock_data      │ Téléchargement cours boursiers historiques (yfinance)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  preprocess_stock_data  │ Calcul indicateurs techniques (SMA, RSI, MACD, etc.) & Target J+1
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│       train_models      │ Entraînement avec validation croisée temporelle (TimeSeriesSplit)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   predict_stock_trends  │ Inférence sur la dernière séance de marché
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   recommend_portfolio   │ Génération des signaux (Buy/Sell/Hold) et niveaux de confiance
└─────────────────────────┘
```

---

## 3. Structure du Répertoire

```text
stock_market_predictions/
├── .astro/                           # Configuration et règles d'intégrité Astronomer
│   ├── config.yaml                   # Métadonnées du projet Astro
│   └── test_dag_integrity_default.py # Test d'intégrité par défaut des DAGs
├── dags/                             # Définition des DAGs Apache Airflow
│   ├── .airflowignore                # Fichiers ignorés par le planificateur Airflow
│   ├── stock_prediction_xgboost.py   # Pipeline complet XGBoost avec GridSearchCV
│   ├── stock_prediction_random_forest.py # Pipeline complet Random Forest
│   └── stock_prediction_logistic_regression.py # Pipeline complet Régression Logistique
├── include/                          # Code métier réutilisable et logique ML
│   ├── config.py                     # Paramètres globaux, liste des actions, chemins
│   ├── fetch_stock_data.py           # Module de téléchargement yfinance
│   ├── preprocess_data.py            # Feature engineering et indicateurs boursiers
│   └── evaluation.py                 # Moteur de décision et règles d'arbitrage
├── tests/                            # Tests automatisés
│   ├── dags/
│   │   └── test_dag_example.py       # Tests d'intégrité (tags, retries, imports Airflow)
│   └── test_pipeline.py              # Tests unitaires du prétraitement et de la décision
├── airflow_settings.yaml             # Variables et connexions Airflow locales
├── Dockerfile                        # Image Astro Runtime basée sur Apache Airflow
├── requirements.txt                  # Dépendances Python (xgboost, scikit-learn, ta, etc.)
├── packages.txt                      # Dépendances système OS (si nécessaire)
├── .env.example                      # Exemple de variables d'environnement
├── .gitignore                        # Fichiers et artefacts ignorés par Git
└── README.md                         # Documentation complète du projet
```

---

## 4. Pipelines de Machine Learning (DAGs)

Le projet propose **trois pipelines indépendants et complémentaires**, permettant de comparer les performances des algorithmes :

### 1. `stock_prediction_xgboost_pipeline`
- **Algorithme** : `XGBClassifier` (Gradient Boosted Trees).
- **Optimisation** : `GridSearchCV` combiné avec `TimeSeriesSplit(n_splits=5)`.
- **Atouts** : Capture les relations non-linéaires complexes et les interactions d'indicateurs.
- **Artefacts** : `{ticker}_xgboost_model.pkl`, `{ticker}_xgboost_metrics.csv`, `stock_signals_xgboost_{date}.csv`.

### 2. `stock_prediction_random_forest_pipeline`
- **Algorithme** : `RandomForestClassifier` (Bagging d'arbres décisionnels).
- **Validation** : `TimeSeriesSplit(n_splits=5)`.
- **Atouts** : Forte robustesse face au surapprentissage (overfitting) et stabilité sur les séries financières bruitées.
- **Artefacts** : `{ticker}_rf_model.pkl`, `{ticker}_rf_metrics.csv`, `stock_signals_rf_{date}.csv`.

### 3. `stock_prediction_logistic_regression_pipeline`
- **Algorithme** : `Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])`.
- **Validation** : `TimeSeriesSplit(n_splits=5)`.
- **Atouts** : Modèle de référence (*baseline*) rapide, interprétable et régularisé.
- **Artefacts** : `{ticker}_logistic_model.pkl`, `{ticker}_logistic_metrics.csv`, `stock_signals_logistic_{date}.csv`.

---

## 5. Feature Engineering & Indicateurs Techniques

Toutes les caractéristiques sont calculées de manière centralisée dans `include/preprocess_data.py` :

| Caractéristique | Type | Description |
| :--- | :--- | :--- |
| `Close` | Prix | Cours de clôture de la séance. |
| `Volume` | Liquidité | Nombre de titres échangés pendant la séance. |
| `SMA_5` | Tendance courte | Moyenne mobile simple sur 5 jours ouvrés (1 semaine de trading). |
| `SMA_20` | Tendance moyenne | Moyenne mobile simple sur 20 jours ouvrés (1 mois de trading). |
| `RSI` | Momentum | *Relative Strength Index* (14 périodes) mesurant la survente (<30) ou surachat (>70). |
| `MACD` | Tendance & Signal | Écart entre l'indicateur MACD et sa ligne de signal. |
| `Daily_Return` | Rendement | Variation relative en pourcentage du cours : $\frac{Close_t - Close_{t-1}}{Close_{t-1}}$. |
| `Volatility` | Risque | Écart-type glissant sur 5 jours normalisé par le cours de clôture. |
| `Momentum` | Dynamique | Écart entre le cours actuel et sa moyenne glissante sur 10 jours. |
| **`Target`** | **Cible (y)** | **$1$ si $Close_{t+1} > Close_t$, sinon $0$.** |

---

## 6. Moteur de Décision & Signaux de Portefeuille

Dans le trading réel, l'accuracy globale ne suffit pas : prédire une hausse avec une faible précision entraîne des pertes en capital. Le moteur de décision ([include/evaluation.py](include/evaluation.py)) applique donc des conditions strictes sur la **Précision directionnelle** et l'**AUC-ROC** :

```text
SI (Precision_UP > 60%) ET (Precision_UP - Precision_DOWN > 10%) ET (Accuracy > 55%) ET (AUC > 0.55) :
    ACTION = "Buy"
    CONFIANCE = "High" si (Precision_UP > 70% et Acc > 65% et AUC > 0.65) sinon "Medium"

SINON SI (Precision_DOWN > 60%) ET (Precision_DOWN - Precision_UP > 10%) ET (Accuracy > 55%) ET (AUC > 0.55) :
    ACTION = "Sell"
    CONFIANCE = "High" si (Precision_DOWN > 70% et Acc > 65% et AUC > 0.65) sinon "Medium"

SINON :
    ACTION = "Hold"
    CONFIANCE = "Low"
```

Les signaux sont sauvegardés chaque jour au format CSV avec la structure suivante :
```csv
Date,Ticker,Model,Prediction,Action,Confidence,Precision_Up,Precision_Down,Accuracy,ROC_AUC
2026-11-13,AAPL,XGBoost,Up,Buy,High,0.732,0.485,0.680,0.710
2026-11-13,TSLA,XGBoost,Down,Sell,Medium,0.420,0.640,0.610,0.620
2026-11-13,MSFT,XGBoost,Up,Hold,Low,0.540,0.510,0.530,0.520
```

---

## 7. Bonnes Pratiques MLOps Appliquées

1. **Aucune fuite de données (*Data Leakage*)** :
   - Remplacement de `train_test_split` par `TimeSeriesSplit`, respectant l'ordre chronologique des séries temporelles boursières.
   - Les indicateurs techniques sont calculés strictement sur les données passées avant de décaler la cible temporelle.
2. **Gestion robuste et multi-plateforme des chemins** :
   - Utilisation de `include/config.py` avec `get_storage_path()` pour garantir la compatibilité entre Windows, macOS, Linux et les conteneurs Docker Airflow (`AIRFLOW_DATA_DIR`).
3. **Persistance cohérente des modèles** :
   - Sérialisation et désérialisation homogènes via `joblib` (`.pkl`), éliminant les bugs de chargement de modèles Keras incompatibles.
4. **Standardisation et conformité Airflow** :
   - Tous les DAGs définissent un dictionnaire `default_args` avec `retries >= 2` et des `tags` explicites pour satisfaire les tests d'intégrité Astronomer.

---

## 8. Installation et Guide de Démarrage

### Prérequis
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) en cours d'exécution.
- [Astro CLI](https://www.astronomer.io/docs/astro/cli/install-cli/) installé sur votre machine.

### 1. Démarrer le cluster Airflow en local
À la racine du projet, lancez :
```bash
astro dev start
```
Cette commande démarre automatiquement les 5 conteneurs Docker :
- Le planificateur (*Scheduler*)
- La base de métadonnées (*Postgres*)
- Le processeur de DAGs (*DAG Processor*)
- Le serveur d'interface web (*API Server / Webserver*)
- Le déclencheur (*Triggerer*)

### 2. Accéder à l'interface Airflow
Ouvrez votre navigateur sur : **[http://localhost:8080](http://localhost:8080)**  
- **Identifiant par défaut** : `admin`
- **Mot de passe par défaut** : `admin`

Vous y retrouverez les 3 DAGs prêts à être activés ou déclenchés manuellement :
- `stock_prediction_xgboost_pipeline`
- `stock_prediction_random_forest_pipeline`
- `stock_prediction_logistic_regression_pipeline`

### 3. Consulter les résultats
Les modèles entraînés et les signaux quotidiens sont stockés dans le dossier configuré par `AIRFLOW_DATA_DIR` (par défaut `/tmp/stock_market_predictions` dans le conteneur).

Pour récupérer le fichier des signaux du jour depuis le conteneur :
```bash
astro dev bash -c "cat /tmp/stock_market_predictions/stock_signals_xgboost_\$(date +%F).csv"
```

---

## 9. Tests & Validation

### Validation de l'intégrité des DAGs Airflow
Exécutez la suite de tests Astro :
```bash
astro dev pytest
```
Ces tests vérifient :
- L'absence d'erreurs d'importation dans les fichiers Python de `dags/`.
- La présence obligatoire de `tags` sur chaque DAG.
- La configuration de `retries >= 2` sur toutes les tâches.

### Tests unitaires locaux (logique de prétraitement et décision)
Si un environnement Python est disponible :
```bash
pytest tests/test_pipeline.py -v
```

---

## 10. Roadmap & Évolutions Futures

- [ ] **Tableau de bord Streamlit** : Interface visuelle interactive affichant la performance historique des signaux et l'évolution du portefeuille simulé.
- [ ] **Stockage persistant Cloud** : Migration des CSV temporaires vers un bucket AWS S3 / Google Cloud Storage ou une base PostgreSQL.
- [ ] **Alertes en temps réel** : Envoi automatique du récapitulatif des signaux par notification Slack / Discord / Telegram chaque matin avant l'ouverture des marchés.
- [ ] **MLflow Model Registry** : Suivi des expériences, tracking des hyperparamètres et gestion du cycle de vie des modèles.
