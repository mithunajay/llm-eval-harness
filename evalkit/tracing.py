"""Step-level tracing. Every step records latency, tokens, estimated cost, and any error,
so a bad output can be traced to the exact step that broke."""
import json, time
from contextlib import contextmanager


def _short(value, limit=300):
    try:
        text = json.dumps(value, default=str)
    except TypeError:
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


class SpanHandle:
    def __init__(self, record):
        self._r = record

    def set_output(self, output):
        self._r["output"] = _short(output)

    def set_tokens(self, tokens_in, tokens_out):
        self._r["tokens_in"], self._r["tokens_out"] = int(tokens_in), int(tokens_out)


class Tracer:
    """rates are (input, output) USD per 1M tokens. They are estimates you configure."""

    def __init__(self, trace_id, rates=(3.0, 15.0)):
        self.trace_id, self.rates, self.spans = trace_id, rates, []
        self._t0 = time.perf_counter()

    @contextmanager
    def span(self, name, **inputs):
        record = {"trace_id": self.trace_id, "span": name, "input": _short(inputs), "output": None,
                  "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "error": None,
                  "start_ms": round((time.perf_counter() - self._t0) * 1000, 2), "latency_ms": None}
        start = time.perf_counter()
        try:
            yield SpanHandle(record)
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            record["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
            record["cost_usd"] = round((record["tokens_in"] * self.rates[0] + record["tokens_out"] * self.rates[1]) / 1e6, 6)
            self.spans.append(record)

    @property
    def total_cost(self):
        return round(sum(s["cost_usd"] for s in self.spans), 6)

    @property
    def total_latency_ms(self):
        return round(sum(s["latency_ms"] or 0 for s in self.spans), 2)

    def failed_step(self):
        for s in self.spans:
            if s["error"]:
                return s["span"]
        return None

    def dump(self, path):
        with open(path, "a") as f:
            for s in self.spans:
                f.write(json.dumps(s) + "\n")
