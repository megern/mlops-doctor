"""Independent regression for observed missing-record loss in CSV import."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.mark.parametrize("csv_text, expected", [
    ('id,value\n,\na,1', [{"id":"", "value":""}, {"id":"a", "value":"1"}]),
    ('id,value\n"",""\na,1', [{"id":"", "value":""}, {"id":"a", "value":"1"}]),
    ('id,value\n\na,1\n\n', [{"id":"a", "value":"1"}]),
    ('value\n""\n1', [{"value":""}, {"value":"1"}]),
])
def test_independent_preserve_explicit_empty_records(csv_text, expected):
    response=TestClient(app).post('/csv/preview', json={"csv_text":csv_text})
    assert response.status_code==200, response.text
    assert response.json()['row_count']==len(expected)
    assert response.json()['preview']==expected

def test_independent_missing_id_is_not_silently_dropped():
    response=TestClient(app).post('/csv/analyze', json={"train_csv":"id,x,y\n,,\na,1,0", "validation_csv":"id,x,y\nb,2,0", "schema":{"id":"string", "x":"number", "y":"number"}, "id_column":"id", "target":"y", "features":["x"]})
    assert response.status_code==422, response.text
    assert "non-null" in response.json()['detail']
