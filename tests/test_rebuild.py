from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)

def test_task_workbench_and_bundled_assets():
    page = client.get('/')
    assert page.status_code == 200
    assert 'Load synthetic example' in page.text
    assert 'Export report' in page.text
    for path in ['/assets/app.js', '/assets/common.js', '/assets/styles.css', '/guide']:
        assert client.get(path).status_code == 200
    assert 'Intended use' in client.get('/guide').text
    assert app.version == '0.2.0'

from app.csv_import import parse_csv
import pytest

def test_quoted_csv_and_bom_preserve_text():
    headers, rows = parse_csv('\ufeffid,note\r\n001,"comma, and ""quoted"" text"\r\n')
    assert headers == ['id', 'note']
    assert rows == [{'id': '001', 'note': 'comma, and "quoted" text'}]

def test_csv_rejects_duplicate_headers_ragged_records_and_invalid_quotes():
    for text in ['a, a\n1,2', 'a,b\n1', 'a,b\n"unterminated,2', 'a,b\n']:
        assert client.post('/csv/preview', json={'csv_text': text}).status_code == 422

def test_csv_limits_and_inert_text_preview():
    assert client.post('/csv/preview', json={'csv_text': 'a\n'+('x'*10001)}).status_code == 422
    assert client.post('/csv/preview', json={'csv_text': 'a\n'+'1\n'*10001}).status_code == 422
    result = client.post('/csv/preview', json={'csv_text': 'value\n<script>alert(1)</script>\n=SUM(A1:A2)'}).json()
    assert result['preview'][0]['value'] == '<script>alert(1)</script>'
    assert result['preview'][1]['value'] == '=SUM(A1:A2)'
    assert result['row_count'] == 2

def test_csv_diagnosis_preserves_string_ids_and_flags_leakage_and_drift():
    payload={'train_csv':'id,x,y\n001,1,0\n002,3,1','validation_csv':'id,x,y\n001,8,1','schema':{'id':'string','x':'number','y':'number'},'id_column':'id','target':'y','features':['x']}
    response=client.post('/csv/analyze',json=payload)
    assert response.status_code==200
    result=response.json()
    assert {f['code'] for f in result['findings']}=={'SPLIT_LEAKAGE','MEAN_DRIFT'}
    assert result['numeric_drift'][0]['standardized_mean_shift']==6

def test_invalid_numeric_csv_cells_are_diagnosed_not_imputed():
    data={'train_csv':'id,x,y\na,text,0\nb,3,1','validation_csv':'id,x,y\nc,2,0','schema':{'id':'string','x':'number','y':'number'},'id_column':'id','target':'y','features':['x']}
    r=client.post('/csv/analyze',json=data).json()
    assert any(f['code']=='SCHEMA_MISMATCH' for f in r['findings'])
    assert r['numeric_drift']==[]
    data['train_csv']='id,x,y\na,NaN,0'
    assert client.post('/csv/analyze',json=data).status_code==422
