from fastapi.testclient import TestClient

from mlops_serving_starter.api.app import create_app


class _FakeService:
    def predict(self, records):
        return [1.0 for _ in records]


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr(
        "mlops_serving_starter.api.app.get_model_service",
        lambda: _FakeService(),
    )
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={
            "records": [
                {
                    "sepal length (cm)": 5.1,
                    "sepal width (cm)": 3.5,
                    "petal length (cm)": 1.4,
                    "petal width (cm)": 0.2,
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {"predictions": [1.0]}

