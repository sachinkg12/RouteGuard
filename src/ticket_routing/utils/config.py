"""Typed experiment configuration loaded from YAML.

Pydantic models keep the surface area small but typed. Adding new fields is
strictly additive; nothing in this module dictates evaluator behavior.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    dataset_name: str
    dataset_path: Optional[str] = None
    text_column: str = "text"
    label_column: str = "label"
    minimum_class_count: int = 10
    max_test_samples_for_llm: int = 2000
    # Bounds API spend during isotonic calibration on the calibration split.
    # 0 disables LLM calibration; the calibration split is otherwise fully used
    # by non-LLM predictors.
    max_calib_samples_for_llm: int = 500
    random_seed: int = 42

    # synthetic loader knobs (ignored unless dataset_name == "synthetic")
    synthetic_n_per_class: int = 80


class SplitConfig(BaseModel):
    train_fraction: float = 0.70
    calibration_fraction: float = 0.10
    test_fraction: float = 0.20


class ClassicalModelConfig(BaseModel):
    enabled: bool = True
    name: str = "tfidf_logreg"
    max_features: int = 50_000
    ngram_max: int = 2
    C: float = 1.0


class ExtraPredictorConfig(BaseModel):
    """Additional predictor instantiated via the registry.

    `kind` is the registry key (e.g. ``tfidf_svm``, ``tfidf_random_forest``,
    ``distilbert``). ``name`` defaults to ``kind`` and is used as the
    experiment label in result tables. ``params`` is forwarded as ``**kwargs``
    to the predictor constructor.

    Predictors registered here are trained on the train split alongside the
    primary classical baseline and evaluated on the (possibly LLM-subsampled)
    test split just like any other predictor.
    """

    enabled: bool = True
    kind: str
    name: Optional[str] = None
    params: dict = Field(default_factory=dict)


class LLMClientConfig(BaseModel):
    # "mock" (default), "openai_compatible", "local_transformers"
    kind: str = "mock"
    model: str = "mock-classifier"
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 128
    request_timeout_s: float = 30.0
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    # Optional provider-specific body params passed straight through to the
    # chat.completions call (e.g. {"chat_template_kwargs": {"enable_thinking": false}}
    # to disable a provider's default reasoning template). Empty = unused.
    extra_body: Optional[dict] = None


class LLMExperimentConfig(BaseModel):
    enabled: bool = False
    name: str = "llm_zero_shot"
    setting: str = "zero_shot"  # zero_shot | few_shot
    k_shot_per_class: int = 3
    client: LLMClientConfig = Field(default_factory=LLMClientConfig)
    self_consistency_samples: int = 1  # >1 enables SelfConsistencyConfidence
    # Optional per-experiment override; falls back to top-level cfg.prompt_template_version.
    prompt_template_version: Optional[str] = None


class ConfidenceConfig(BaseModel):
    methods: List[str] = Field(default_factory=lambda: ["model_reported"])


class AbstentionConfig(BaseModel):
    thresholds: List[float] = Field(
        default_factory=lambda: [0.50, 0.60, 0.70, 0.80, 0.90]
    )
    agreement_min: int = 2


class CostConfig(BaseModel):
    correct_auto_route: float = 0.0
    human_triage: float = 1.0
    wrong_auto_route_options: List[float] = Field(default_factory=lambda: [2.0, 5.0, 10.0])
    default_wrong_auto_route: float = 5.0


class BootstrapConfig(BaseModel):
    enabled: bool = False
    samples: int = 1000
    seed: int = 1337


class ExperimentConfig(BaseModel):
    run_id: str = "default"
    output_dir: str = "outputs"
    dataset: DatasetConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    classical: ClassicalModelConfig = Field(default_factory=ClassicalModelConfig)
    extra_predictors: List[ExtraPredictorConfig] = Field(default_factory=list)
    llm_experiments: List[LLMExperimentConfig] = Field(default_factory=list)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    prompt_template_version: str = "v1"


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    return ExperimentConfig(**raw)


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
