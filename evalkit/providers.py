"""Model providers behind one small interface. MockProvider simulates a model so the harness
can be tested offline and deterministically. AnthropicProvider calls the real API."""
import asyncio, json, os, re, time, urllib.error, urllib.request
from dataclasses import dataclass


class ProviderError(Exception):
    pass


@dataclass
class Completion:
    text: str
    tokens_in: int
    tokens_out: int


class MockProvider:
    """mode 'good' writes grounded openers. mode 'regressed' simulates a bad prompt/model change:
    invents numbers on cases with an odd repo count, and rambles on large accounts."""

    def __init__(self, mode="good", delay=0.0):
        self.mode, self.delay, self.name = mode, delay, f"mock:{mode}"

    async def complete(self, prompt, system=""):
        if self.delay:
            await asyncio.sleep(self.delay)
        m = re.search(r"^FACTS: (\{.*\})$", prompt, re.M)
        if not m:
            raise ProviderError("mock provider could not find FACTS in prompt")
        f = json.loads(m.group(1))
        org, repo, active, stars = f["org"], f["top_repo"], f["active_repos"], f["total_stars"]
        text = f"{org} has {active} repos with recent commits, led by {repo}."
        if self.mode == "regressed":
            if active % 2 == 1:
                text = f"{org} grew 300% this year, and {repo} is a game-changer."
            elif stars > 10000:
                text = f"{org} is clearly doing a great deal of interesting work across " + "many " * 40 + f"areas, led by {repo}."
        return Completion(text, len(prompt.split()), len(text.split()))


class MockJudgeProvider:
    """Deterministic stand-in for an LLM judge. flaky=True returns unparseable text every 3rd call
    so the harness's handling of judge failures can be tested."""

    def __init__(self, flaky=False):
        self.flaky, self.calls, self.name = flaky, 0, "mock-judge"

    async def complete(self, prompt, system=""):
        self.calls += 1
        if self.flaky and self.calls % 3 == 0:
            return Completion("Sure! I think this looks fine overall.", 50, 8)
        facts = json.loads(re.search(r"^FACTS: (\{.*\})$", prompt, re.M).group(1))
        output = re.search(r"^OUTPUT: (.*)$", prompt, re.M).group(1)
        from evalkit.checks import banned_phrases, numbers_grounded, word_limit
        case = {"facts": facts, "max_words": 35}
        good = (facts["top_repo"].lower() in output.lower() and numbers_grounded(case, output).passed
                and banned_phrases(case, output).passed and word_limit(case, output).passed)
        return Completion(json.dumps({"score": 5 if good else 2, "reason": "grounded" if good else "ungrounded or off-style"}), 60, 12)


class ScriptedProvider:
    """Calls fn(prompt) -> text (or an Exception to raise). For tests and demos of any task."""

    def __init__(self, fn, name="scripted"):
        self.fn, self.name = fn, name

    async def complete(self, prompt, system=""):
        out = self.fn(prompt)
        if isinstance(out, Exception):
            raise out
        return Completion(out, len(prompt.split()), len(out.split()))


RETRY_CODES = (429, 500, 502, 503, 529)


class AnthropicProvider:
    """Real API call (stdlib only, run in a thread so it does not block the event loop).
    Needs ANTHROPIC_API_KEY. Retries rate-limit and server errors with exponential backoff.
    Temperature defaults to 0 for repeatable evals; set EVALKIT_TEMPERATURE to change it, or to an
    empty string to omit it if your model rejects the parameter."""

    def __init__(self, model=None, max_tokens=400, timeout=60, retries=3, opener=None, sleep=None):
        self.model = model or os.environ.get("EVALKIT_MODEL", "claude-sonnet-5-5")
        self.max_tokens, self.timeout, self.retries = max_tokens, timeout, retries
        self._open = opener or urllib.request.urlopen
        self._sleep = sleep or time.sleep
        self.name = f"anthropic:{self.model}"

    def _call(self, prompt, system):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        payload = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                   "messages": [{"role": "user", "content": prompt}]}
        temp = os.environ.get("EVALKIT_TEMPERATURE", "0")
        if temp != "":
            payload["temperature"] = float(temp)
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(payload).encode(),
                                     method="POST", headers={"content-type": "application/json", "x-api-key": key,
                                                             "anthropic-version": "2023-06-01"})
        attempt = 0
        while True:
            try:
                with self._open(req, timeout=self.timeout) as r:
                    data = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code in RETRY_CODES and attempt < self.retries:
                    self._sleep(2 ** attempt)
                    attempt += 1
                    continue
                raise ProviderError(f"HTTP {e.code}")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                raise ProviderError(f"{type(e).__name__}: {e}")
        text = "".join(b.get("text", "") for b in data.get("content", []))
        u = data.get("usage", {})
        return Completion(text.strip(), u.get("input_tokens", 0), u.get("output_tokens", 0))

    async def complete(self, prompt, system=""):
        return await asyncio.to_thread(self._call, prompt, system)


def make_provider(spec):
    if spec.startswith("mock:"):
        return MockProvider(spec.split(":", 1)[1])
    if spec == "anthropic":
        return AnthropicProvider()
    raise ValueError(f"unknown provider {spec!r}; use mock:good, mock:regressed, or anthropic")
