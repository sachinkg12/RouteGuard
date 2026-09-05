"""Classical baseline 3: TF-IDF + RandomForest.

Adds a non-linear tree-ensemble baseline. Defaults are sklearn defaults
plus n_estimators=300 (cheap on a corpus of this size) and n_jobs=-1.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from .base import PARSE_OK, PredictionBatch, Predictor
from .registry import register


@register("tfidf_random_forest")
class TfidfRandomForestPredictor(Predictor):
    name = "tfidf_random_forest"

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_max: int = 2,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.max_features = max_features
        self.ngram_max = ngram_max
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.pipeline: Pipeline | None = None
        self._classes_: List[str] = []

    def fit(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[str],
        label_names: Sequence[str],
    ) -> None:
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.max_features,
                        ngram_range=(1, self.ngram_max),
                        min_df=1,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.pipeline.fit(list(train_texts), list(train_labels))
        self._classes_ = list(self.pipeline.named_steps["clf"].classes_)

    def predict(self, texts: Sequence[str]) -> PredictionBatch:
        if self.pipeline is None:
            raise RuntimeError("predict() called before fit().")
        texts = list(texts)
        proba = self.pipeline.predict_proba(texts)
        top_idx = np.argmax(proba, axis=1)
        predicted_labels = [self._classes_[i] for i in top_idx]
        confidence = proba[np.arange(len(texts)), top_idx].tolist()
        return PredictionBatch(
            predicted_labels=predicted_labels,
            confidence_scores=confidence,
            raw_outputs=None,
            parse_status=[PARSE_OK] * len(texts),
            latency_seconds=None,
            cost_estimate=[0.0] * len(texts),
            model_metadata={
                "predictor": self.name,
                "max_features": self.max_features,
                "ngram_max": self.ngram_max,
                "n_estimators": self.n_estimators,
                "classes": self._classes_,
            },
        )

    def predict_proba_full(self, texts: Sequence[str]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("predict_proba_full() called before fit().")
        return self.pipeline.predict_proba(list(texts))
