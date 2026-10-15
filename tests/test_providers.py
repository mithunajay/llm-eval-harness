import io, json, os, urllib.error
from evalkit.providers import AnthropicProvider, ProviderError, ScriptedProvider
import asyncio


class FakeOpener:
    def __init__(self, script):
        self.script, self.calls, self.last_body = list(script), 0, None

    def __call__(self, req, timeout=None):
        self.calls += 1
        self.last_body = json.loads(req.data)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return io.BytesIO(json.dumps(item).encode())


def http_error(code):
    return urllib.error.HTTPError("https://api.anthropic.com", code, "err", {}, None)


OK = {"content": [{"type": "text", "text": " hello "}], "usage": {"input_tokens": 11, "output_tokens": 3}}


def with_key(fn):
    old = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = old


def test_retries_rate_limits_with_backoff_then_succeeds():
    sleeps, op = [], FakeOpener([http_error(429), http_error(503), OK])
    p = AnthropicProvider(opener=op, sleep=sleeps.append)
    comp = with_key(lambda: p._call("hi", "sys"))
    assert comp.text == "hello" and (comp.tokens_in, comp.tokens_out) == (11, 3)
    assert op.calls == 3 and sleeps == [1, 2]


def test_gives_up_after_retries_are_exhausted():
    op = FakeOpener([http_error(429)] * 4)
    p = AnthropicProvider(opener=op, sleep=lambda s: None, retries=3)
    try:
        with_key(lambda: p._call("hi", "sys"))
    except ProviderError as e:
        assert "429" in str(e) and op.calls == 4
    else:
        raise AssertionError("expected ProviderError")


def test_client_errors_are_not_retried():
    op = FakeOpener([http_error(400)])
    p = AnthropicProvider(opener=op, sleep=lambda s: None)
    try:
        with_key(lambda: p._call("hi", "sys"))
    except ProviderError as e:
        assert "400" in str(e) and op.calls == 1
    else:
        raise AssertionError("expected ProviderError")


def test_missing_key_is_a_clear_error():
    old = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        AnthropicProvider(opener=FakeOpener([OK]))._call("hi", "sys")
    except ProviderError as e:
        assert "ANTHROPIC_API_KEY" in str(e)
    else:
        raise AssertionError("expected ProviderError")
    finally:
        if old is not None:
            os.environ["ANTHROPIC_API_KEY"] = old


def test_temperature_defaults_to_zero_and_can_be_omitted():
    op = FakeOpener([OK, OK])
    p = AnthropicProvider(opener=op, sleep=lambda s: None)
    with_key(lambda: p._call("hi", "sys"))
    assert op.last_body["temperature"] == 0.0 and op.last_body["messages"][0]["content"] == "hi"
    os.environ["EVALKIT_TEMPERATURE"] = ""
    try:
        with_key(lambda: p._call("hi", "sys"))
    finally:
        del os.environ["EVALKIT_TEMPERATURE"]
    assert "temperature" not in op.last_body


def test_scripted_provider_can_raise():
    p = ScriptedProvider(lambda prompt: ProviderError("nope"))
    try:
        asyncio.run(p.complete("x"))
    except ProviderError:
        return
    raise AssertionError("expected ProviderError")
