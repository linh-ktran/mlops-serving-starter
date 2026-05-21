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
    monkeypatch.setenv("TARGET_NAME", "afrr_capacity_price_up")
    monkeypatch.setenv("FORECAST_HORIZON", "1")
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={
            "records": [
                {
                    "afrr_capacity_price_up_lag_1_h1": 25.0,
                    "rolling_mean_7": 23.0,
                    "holiday_status": 0,
                    "weekend_status": 0,
                }
            ]
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["predictions"] == [1.0]
    assert data["target"] == "afrr_capacity_price_up"
    assert data["horizon"] == 1
    assert data["unit"] == "EUR/MW"
