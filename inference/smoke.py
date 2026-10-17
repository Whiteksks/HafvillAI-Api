"""Check a running API with real adapters without downloading or changing weights."""

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


def request(base_url, path, api_key, timeout, payload=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    with urlopen(Request(base_url.rstrip("/") + path, data=data, headers=headers), timeout=timeout) as response:
        return json.load(response)


def main():
    load_dotenv(Path(__file__).parent / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--role", choices=["consciousness", "subconsciousness", "all"], default="all")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    key = os.getenv("HAFVILLAI_API_KEY", "")
    health = request(args.base_url, "/health", key, args.timeout)
    if health.get("status") != "ready":
        raise RuntimeError("API не готов")
    roles = health["roles"] if args.role == "all" else [args.role]
    examples = json.loads((Path(__file__).resolve().parent / "examples/requests.json").read_text(encoding="utf-8"))
    for role in roles:
        if role not in health["roles"]:
            raise RuntimeError(f"{role} отключена")
        payload = {"messages": examples[role], "max_new_tokens": args.max_new_tokens}
        path = "/chat" if role == "consciousness" else "/summarize"
        first = request(args.base_url, path, key, args.timeout, payload)
        second = request(args.base_url, path, key, args.timeout, payload)
        expected_step = health["adapter_steps"][role]
        if first["role"] != role or first["usage"]["completion_tokens"] < 1 or first["adapter_step"] != expected_step:
            raise RuntimeError(f"{role}: неверный ответ API")
        if (first["content"], first["finish_reason"], first["usage"]) != (
            second["content"], second["finish_reason"], second["usage"],
        ):
            raise RuntimeError(f"{role}: повторный greedy-запрос дал другой результат")
        print(json.dumps(first, ensure_ascii=False, indent=2))
        print(f"OK: {role}, checkpoint={expected_step}, два независимых greedy-запроса совпали")


if __name__ == "__main__":
    try:
        main()
    except HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from None
    except (URLError, TimeoutError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
