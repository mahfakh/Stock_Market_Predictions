"""
Tests d'intégrité des DAGs Airflow.
Vérifie qu'aucun DAG ne comporte d'erreur d'importation, que les tags sont présents,
et que le nombre de retries est >= 2 (exigence Astronomer).
"""

import os
import logging
from contextlib import contextmanager
import pytest

try:
    from airflow.models import DagBag
    AIRFLOW_INSTALLED = True
except ImportError:
    AIRFLOW_INSTALLED = False


@contextmanager
def suppress_logging(namespace):
    logger = logging.getLogger(namespace)
    old_value = logger.disabled
    logger.disabled = True
    try:
        yield
    finally:
        logger.disabled = old_value


def get_import_errors():
    if not AIRFLOW_INSTALLED:
        return [(None, None)]
    with suppress_logging("airflow"):
        dag_bag = DagBag(include_examples=False)

        def strip_path_prefix(path):
            airflow_home = os.environ.get("AIRFLOW_HOME", "")
            return os.path.relpath(path, airflow_home) if airflow_home else path

        return [(None, None)] + [
            (strip_path_prefix(k), v.strip()) for k, v in dag_bag.import_errors.items()
        ]


def get_dags():
    if not AIRFLOW_INSTALLED:
        return []
    with suppress_logging("airflow"):
        dag_bag = DagBag(include_examples=False)

    def strip_path_prefix(path):
        airflow_home = os.environ.get("AIRFLOW_HOME", "")
        return os.path.relpath(path, airflow_home) if airflow_home else path

    return [(k, v, strip_path_prefix(v.fileloc)) for k, v in dag_bag.dags.items()]


@pytest.mark.parametrize(
    "rel_path,rv", get_import_errors(), ids=[x[0] for x in get_import_errors()]
)
def test_file_imports(rel_path, rv):
    """Vérifie l'absence d'erreurs d'importation dans les DAGs."""
    if not AIRFLOW_INSTALLED:
        pytest.skip("Airflow non installé sur l'hôte (s'exécute dans le conteneur Astro)")
    if rel_path and rv:
        raise Exception(f"{rel_path} a échoué à l'importation avec : {rv}")


@pytest.mark.parametrize(
    "dag_id,dag,fileloc", get_dags(), ids=[x[2] for x in get_dags()] if get_dags() else ["no_dag"]
)
def test_dag_tags(dag_id, dag, fileloc):
    """Vérifie que chaque DAG possède au moins un tag."""
    if not AIRFLOW_INSTALLED:
        pytest.skip("Airflow non installé sur l'hôte")
    assert dag.tags, f"{dag_id} dans {fileloc} n'a aucun tag défini."


@pytest.mark.parametrize(
    "dag_id,dag,fileloc", get_dags(), ids=[x[2] for x in get_dags()] if get_dags() else ["no_dag"]
)
def test_dag_retries(dag_id, dag, fileloc):
    """Vérifie que chaque DAG configure retries >= 2."""
    if not AIRFLOW_INSTALLED:
        pytest.skip("Airflow non installé sur l'hôte")
    retries = dag.default_args.get("retries", None)
    assert retries is not None and retries >= 2, f"{dag_id} dans {fileloc} doit avoir retries >= 2."
