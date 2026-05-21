import pytest
from app.app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test if the health check endpoint returns 200 OK."""
    response = client.get('/health')

    assert response.status_code == 200
    assert response.json == {'status': 'healthy'}


def test_predict_endpoint(client):
    """Test if the predict endpoint accepts form data and returns HTML."""

    mock_form_data = {
        "fixed acidity": "7.4",
        "volatile acidity": "0.7",
        "citric acid": "0.0",
        "residual sugar": "1.9",
        "chlorides": "0.076",
        "free sulfur dioxide": "11.0",
        "total sulfur dioxide": "34.0",
        "density": "0.9978",
        "pH": "3.51",
        "sulphates": "0.56",
        "alcohol": "9.4"
    }

    response = client.post('/predict', data=mock_form_data)

    # Should return a successful HTML page
    assert response.status_code == 200
    assert b"Prediction Result" in response.data
