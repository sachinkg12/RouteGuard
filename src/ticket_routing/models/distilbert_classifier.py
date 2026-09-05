"""Fine-tuned DistilBERT predictor for IT-ticket classification.

A fine-tuned encoder baseline, included to test whether the classical-vs-LLM
gap is explained simply by the LLMs not being fine-tuned.

Requires the ``llm-local`` extra:
    pip install -e .[llm-local]
(brings in ``transformers`` and ``torch``).

Imports are lazy so the rest of the package never pays the cost.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .base import PARSE_OK, PredictionBatch, Predictor
from .registry import register


def _require_transformers():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "DistilBertPredictor requires `transformers` and `torch`. "
            "Install with: pip install -e .[llm-local]"
        ) from e


@register("distilbert")
class DistilBertPredictor(Predictor):
    """Fine-tune ``distilbert-base-uncased`` for sequence classification.

    Trains for ``num_train_epochs`` (default 2) using HuggingFace ``Trainer``.
    Defaults are tuned for ~33k tickets, 8 classes, single GPU or CPU.
    """

    name = "distilbert"

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_train_epochs: int = 2,
        per_device_train_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        max_seq_length: int = 256,
        warmup_ratio: float = 0.06,
        random_state: int = 42,
        fp16: bool = False,
        output_dir: str = ".distilbert_train",
        logging_steps: int = 200,
    ) -> None:
        self.model_name = model_name
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_seq_length = max_seq_length
        self.warmup_ratio = warmup_ratio
        self.random_state = random_state
        self.fp16 = fp16
        self.output_dir = output_dir
        self.logging_steps = logging_steps

        self._tokenizer = None
        self._model = None
        self._label_to_id: dict[str, int] = {}
        self._id_to_label: dict[int, str] = {}
        self._device = None

    def fit(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[str],
        label_names: Sequence[str],
    ) -> None:
        _require_transformers()
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
        from torch.utils.data import Dataset

        set_seed(self.random_state)

        # Stable label encoding using the canonical label_names order.
        self._label_to_id = {lbl: i for i, lbl in enumerate(label_names)}
        self._id_to_label = {i: lbl for lbl, i in self._label_to_id.items()}
        num_labels = len(self._label_to_id)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=self._id_to_label,
            label2id=self._label_to_id,
        )

        train_encodings = self._tokenizer(
            list(train_texts),
            truncation=True,
            padding=False,
            max_length=self.max_seq_length,
        )
        train_label_ids = [self._label_to_id[lbl] for lbl in train_labels]

        class _TicketDataset(Dataset):
            def __init__(self, encodings, label_ids):
                self.encodings = encodings
                self.label_ids = label_ids

            def __len__(self):
                return len(self.label_ids)

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                item["labels"] = torch.tensor(self.label_ids[idx])
                return item

        train_dataset = _TicketDataset(train_encodings, train_label_ids)

        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            logging_steps=self.logging_steps,
            save_strategy="no",
            report_to=[],
            fp16=self.fp16,
            seed=self.random_state,
            dataloader_num_workers=0,
        )

        def _collator(features):
            return self._tokenizer.pad(features, return_tensors="pt")

        trainer = Trainer(
            model=self._model,
            args=args,
            train_dataset=train_dataset,
            data_collator=_collator,
        )
        trainer.train()
        self._model.eval()
        self._device = next(self._model.parameters()).device

    def predict(self, texts: Sequence[str]) -> PredictionBatch:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("predict() called before fit().")
        _require_transformers()
        import torch

        texts_list = list(texts)
        predicted_labels: List[str] = []
        confidences: List[float] = []

        bs = self.per_device_eval_batch_size
        with torch.no_grad():
            for start in range(0, len(texts_list), bs):
                batch = texts_list[start : start + bs]
                encoded = self._tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                ).to(self._device)
                logits = self._model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                top_ids = probs.argmax(axis=1)
                for i, top in enumerate(top_ids):
                    predicted_labels.append(self._id_to_label[int(top)])
                    confidences.append(float(probs[i, top]))

        return PredictionBatch(
            predicted_labels=predicted_labels,
            confidence_scores=confidences,
            raw_outputs=None,
            parse_status=[PARSE_OK] * len(texts_list),
            latency_seconds=None,
            cost_estimate=[0.0] * len(texts_list),
            model_metadata={
                "predictor": self.name,
                "model_name": self.model_name,
                "num_train_epochs": self.num_train_epochs,
                "max_seq_length": self.max_seq_length,
            },
        )

    def predict_proba_full(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("predict_proba_full() called before fit().")
        _require_transformers()
        import torch

        texts_list = list(texts)
        n_labels = len(self._label_to_id)
        out = np.zeros((len(texts_list), n_labels), dtype=float)
        bs = self.per_device_eval_batch_size
        with torch.no_grad():
            for start in range(0, len(texts_list), bs):
                batch = texts_list[start : start + bs]
                encoded = self._tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                ).to(self._device)
                logits = self._model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                out[start : start + len(batch), :] = probs
        return out
