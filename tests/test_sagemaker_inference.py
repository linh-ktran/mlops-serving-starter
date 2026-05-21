import json

import pandas as pd

from mlops_serving_starter.sagemaker.inference import input_fn, output_fn, predict_fn


class _FakeModel:
    def predict(self, frame: pd.DataFrame):
        return [0 for _ in range(len(frame))]


def test_input_fn_json():
    payload = json.dumps({"records": [{"a": 1.0, "b": 2.0}]})
    frame = input_fn(payload, "application/json")

    assert list(frame.columns) == ["a", "b"]
    assert frame.shape == (1, 2)


def test_predict_and_output_json():
    frame = pd.DataFrame([{"a": 1.0, "b": 2.0}])
    prediction = predict_fn(frame, _FakeModel())
    output = output_fn(prediction, "application/json")

    assert output == '{"predictions": [0.0]}'
