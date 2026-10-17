"""Browser test of summaries with temporary files and a mocked OpenAI API."""

import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "test-results"


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as workspace, socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        command = """
import asyncio
from pathlib import Path
import sys
import httpx
import uvicorn
sys.path.insert(0, str(Path.cwd() / "tests"))
from test_summaries import fixture, success
from server import create_app
settings, _, _, _ = fixture(Path(sys.argv[1]) / "data", count=205)
app = create_app(settings)
async def handler(request):
    if request.method == "GET":
        return httpx.Response(200, json={"id": "test-model"})
    dialogue = request.read().decode()
    await asyncio.sleep(.02 if "Dialogue " in dialogue else 1)
    return httpx.Response(200, json=success("Alice entrusted a letter to the librarian. The librarian promised to keep it safe."))
app.state.summaries.transport = httpx.MockTransport(handler)
uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[2]), log_level="error")
"""
        process = subprocess.Popen([sys.executable, "-c", command, workspace, str(port)], cwd=ROOT)
        url = f"http://127.0.0.1:{port}"
        try:
            for _ in range(100):
                try:
                    with urlopen(url + "/api/summaries/status", timeout=1):
                        break
                except OSError:
                    if process.poll() is not None:
                        raise RuntimeError("Test server exited")
                    time.sleep(.1)
            else:
                raise RuntimeError("Test server did not start")
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page(viewport={"width": 1440, "height": 1100}, accept_downloads=True)
                errors = []
                page.on("pageerror", lambda exc: errors.append(str(exc)))
                page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
                page.goto(url + "/#summaries")
                for path in ["/", "/static/summaries.js?v=5"]:
                    assert page.request.get(url + path).headers["cache-control"] == "no-store"
                expect(page.locator("#summaries")).to_be_visible()
                expect(page.locator("#summary-source")).to_contain_text("205 dialogues")
                expect(page.locator("#summary-start")).to_be_enabled()
                expect(page.locator("#summary-key-status")).to_contain_text("Server key configured")
                expect(page.locator("#summary-model")).to_have_value("test-model")
                expect(page.locator("#summary-batch option")).to_have_count(4)
                expect(page.locator("#summary-batch")).to_have_value("1")
                expect(page.locator("#summary-batch")).to_contain_text("101–200")
                expect(page.locator("#summary-concurrency")).to_have_value("4")
                assert "1. Read" in page.locator("#summary-rules").input_value()
                page.locator("#summary-check").click()
                expect(page.locator("#summary-connection")).to_contain_text("Connected")
                page.locator("#summary-limit").fill("1")
                page.locator("#summary-instruction").fill("Summarize the entire conversation.")
                page.locator("#summary-rules").fill("Saved custom rules.")
                page.locator("#summary-start").click()
                expect(page.locator("#summary-status")).to_contain_text("Summary dataset ready", timeout=15000)
                expect(page.locator("#summary-status")).to_contain_text("1 / 1")
                expect(page.locator("#summary-preview")).to_contain_text("Alice entrusted")
                with page.expect_download() as event:
                    page.locator("#summary-download").click()
                record = json.loads(Path(event.value.path()).read_text())
                assert set(record) == {"messages", "example_id", "split", "split_group", "source"}
                assert [m["role"] for m in record["messages"]] == ["system", "user", "assistant"]
                assert record["messages"][0]["content"] == "Summarize the entire conversation."
                assert "USER:\nAlice:" in record["messages"][1]["content"]
                page.reload()
                expect(page.locator("#summary-status")).to_contain_text("1 / 1")
                expect(page.locator("#summary-rules")).to_have_value("Saved custom rules.")
                page.locator("#summary-default-rules").click()
                rules = page.locator("#summary-rules").input_value()
                assert "10. Return only the summary" in rules
                assert "natural, coherent prose" in rules
                assert "doubt is not a final refusal" in rules
                assert "information needed to continue it" in page.locator("#summary-instruction").input_value()
                expect(page.locator("#summary-rules-status")).to_contain_text("Default instruction and rules loaded")
                page.reload()
                expect(page.locator("#summary-rules")).to_have_value("Saved custom rules.")
                expect(page.locator("#summary-instruction")).to_have_value("Summarize the entire conversation.")
                page.locator("#summary-default-rules").click()
                page.locator("#summary-limit").fill("2")
                page.locator("#summary-concurrency").fill("1")
                page.locator("#summary-start").click()
                expect(page.locator("#summary-stop")).to_be_visible()
                expect(page.locator("#summary-default-rules")).to_be_disabled()
                page.locator("#summary-stop").click()
                expect(page.locator("#summary-status")).to_contain_text("Stopped", timeout=15000)
                expect(page.locator("#summary-resume")).to_be_visible()
                page.locator("#summary-resume").click()
                expect(page.locator("#summary-status")).to_contain_text("Summary dataset ready", timeout=15000)
                expect(page.locator("#summary-status")).to_contain_text("2 / 2")
                page.locator("#summary-batch").select_option("2")
                page.locator("#summary-concurrency").fill("4")
                page.locator("#summary-limit").fill("0")
                expect(page.locator("#summary-selection")).to_contain_text("dialogues: 101–200")
                page.locator("#summary-start").click()
                expect(page.locator("#summary-status")).to_contain_text("Summary dataset ready", timeout=15000)
                expect(page.locator("#summary-status")).to_contain_text("100 / 100")
                expect(page.locator("#summary-run")).to_contain_text("batch 2")
                expect(page.locator("#summary-run")).to_contain_text("concurrency: 4")
                with page.expect_download() as event:
                    page.locator("#summary-download").click()
                records = [json.loads(line) for line in Path(event.value.path()).read_text().splitlines()]
                assert len(records) == 100
                assert "information needed to continue it" in records[0]["messages"][0]["content"]
                assert "Dialogue 101:" in records[0]["messages"][1]["content"]
                assert "Dialogue 200:" in records[-1]["messages"][1]["content"]
                page.reload()
                expect(page.locator("#summary-batch")).to_have_value("2")
                expect(page.locator("#summary-concurrency")).to_have_value("4")
                expect(page.locator("#summary-run")).to_contain_text("coser_batch_2_100_")
                page.locator("#summary-merge").click()
                expect(page.locator("#summary-merge-status")).to_contain_text("100 dialogues · batches: 2", timeout=15000)
                with page.expect_download() as event:
                    page.locator("#summary-merged-download").click()
                merged = [json.loads(line) for line in Path(event.value.path()).read_text().splitlines()]
                assert merged == records
                page.locator("#summary-batch").select_option("3")
                page.locator("#summary-start").click()
                expect(page.locator("#summary-merge")).to_be_disabled()
                expect(page.locator("#summary-status")).to_contain_text("5 / 5", timeout=15000)
                page.locator("#summary-merge").click()
                expect(page.locator("#summary-merge-status")).to_contain_text("105 dialogues · batches: 2, 3", timeout=15000)
                with page.expect_download() as event:
                    page.locator("#summary-merged-download").click()
                merged = [json.loads(line) for line in Path(event.value.path()).read_text().splitlines()]
                assert len(merged) == 105
                assert "Dialogue 205:" in merged[-1]["messages"][1]["content"]
                page.reload()
                expect(page.locator("#summary-merged-download")).to_be_visible()
                expect(page.locator("#summary-merge-status")).to_contain_text("105 dialogues · batches: 2, 3")
                page.screenshot(path=str(ARTIFACTS / "summaries-desktop.png"), full_page=True)
                page.set_viewport_size({"width": 390, "height": 844})
                page.screenshot(path=str(ARTIFACTS / "summaries-mobile.png"), full_page=True)
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Mobile overflow"
                assert not errors, errors
                browser.close()
                print("OK: CoSER split provenance, connection, rules, batches, merge, reload, stop/resume, desktop/mobile; no browser errors.")
        finally:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
