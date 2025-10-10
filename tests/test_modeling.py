# -*- coding: utf-8 -*-
import numpy as np

from src.modeling import build_sequence_model
from src.config import SequenceModelSpec


def test_build_sequence_model_trains_and_predicts():
    spec = SequenceModelSpec(
        sequence_len=3,
        num_classes=5,
        embedding_dim=4,
        hidden_units=(8,),
        dropout=0.1,
    )
    model = build_sequence_model(spec, window_size=2, learning_rate=1e-3, name="test_lstm")
    x = np.random.randint(0, spec.num_classes, size=(12, 2, spec.sequence_len))
    y = np.random.randint(0, spec.num_classes, size=(12, spec.sequence_len))
    history = model.fit(x, y, epochs=1, batch_size=4, verbose=0)

    assert "loss" in history.history
    preds = model.predict(x[:2], verbose=0)
    assert preds.shape == (2, spec.sequence_len, spec.num_classes)

