import json
from research_extension.core import clean_log, major_cluster, read_fixtures, read_real_world
from research_extension.providers import parse_json_answer


def test_clean_log_removes_volatile_fields():
    data="2026-07-14T12:45:01Z Error in /home/runner/work/repo/a.py line 123 hash abcdef1234567890"
    clean=clean_log(data)
    assert "123" not in clean and "abcdef1234567890" not in clean
    assert "/home" not in clean and "2026" not in clean
    assert "error" in clean


def test_fixture_evidence_is_complete():
    fixtures=read_fixtures()
    assert len(fixtures)==223
    assert all(f.expected and f.text for f in fixtures)
    assert len(read_real_world())==22


def test_no_majority_with_three_distinct_clusters():
    winner, votes=major_cluster([{"cluster":0},{"cluster":1},{"cluster":2}])
    assert winner is None and votes==1


def test_majority_two_to_one():
    winner, votes=major_cluster([{"cluster":4},{"cluster":2},{"cluster":4}])
    assert winner==4 and votes==2


def test_llm_json_parser_rejects_unlisted_clusters():
    assert parse_json_answer('{"diagnosis":"Version mismatch.","cluster":"Dependency"}',
                             {"Dependency"})["cluster"]=="Dependency"
    try:
        parse_json_answer('{"diagnosis":"Wrong.","cluster":"Invented"}',{"Dependency"})
    except ValueError:
        return
    raise AssertionError("Unlisted category was accepted")


def test_llm_provider_ollama_payload_and_json(monkeypatch):
    from research_extension import providers
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def read(self):return b'{"message":{"content":"{\\"diagnosis\\":\\"Version conflict.\\",\\"cluster\\":\\"Dependencies\\"}"}}'
    def urlopen(request,timeout):
        assert request.full_url.endswith('/api/chat')
        body=json.loads(request.data)
        assert body['model']=='qwen2.5:0.5b'
        assert body['stream'] is False and body['format']=='json'
        return Response()
    monkeypatch.setattr(providers,'urlopen',urlopen)
    output=providers.diagnose('fatal dependency conflict',
                             [{'cluster_name':'Dependencies','score':0.79,'example':'pip failed to resolve'}]*3,
                             ['Dependencies'],backend='ollama',model='qwen2.5:0.5b')
    assert output['cluster']=='Dependencies'


def test_external_api_requires_credentials(monkeypatch):
    from research_extension.providers import chat_llm
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    try:
        chat_llm([],backend='openai',model='gpt-4.1-mini')
    except RuntimeError as e:
        assert 'OPENAI_API_KEY' in str(e)
        return
    raise AssertionError('A missing key was not rejected')
