"""Ollama is started (and the model loaded) by the server itself, so the user never has to run `ollama serve` first."""
import socket, time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import assistant
import server
from test_assistant_and_explain import FakeOllama


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close(); return port


class FakeProc:
    def __init__(self): self.terminated = False
    def terminate(self): self.terminated = True


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    assistant._state.update(phase="idle", message=""); assistant._proc = None
    monkeypatch.delenv("EMOTISENSE_AUTOSTART_OLLAMA", raising=False)
    monkeypatch.setenv("EMOTISENSE_OLLAMA_WAIT", "5")
    yield
    assistant._proc = None


@pytest.fixture
def running(monkeypatch):
    fake = FakeOllama()
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{fake.port}")
    monkeypatch.setattr(assistant, "_spawn", lambda exe: pytest.fail("must not start a second Ollama"))
    yield fake
    fake.close()


def stopped(monkeypatch):
    port = free_port()
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{port}")
    return port


def test_does_nothing_but_warm_up_when_ollama_already_runs(running):
    assistant.bootstrap()
    assert assistant._state["phase"] == "ready"
    assert running.generate_requests == [{"model": "qwen3:8b", "keep_alive": "30m"}]      # model loaded into memory, kept for 30 min


def test_starts_ollama_when_it_is_not_running(monkeypatch):
    port, started, servers = stopped(monkeypatch), [], []
    monkeypatch.setattr(assistant, "find_ollama_executable", lambda: r"C:\Users\x\AppData\Local\Programs\Ollama\ollama.exe")
    def fake_spawn(exe):
        started.append(exe); servers.append(FakeOllama(port)); return FakeProc()          # "ollama serve" begins listening
    monkeypatch.setattr(assistant, "_spawn", fake_spawn)
    assert assistant.is_up() is False
    assistant.bootstrap()
    try:
        assert started == [r"C:\Users\x\AppData\Local\Programs\Ollama\ollama.exe"]
        assert assistant._state["phase"] == "ready" and assistant.is_up()
        assert servers[0].generate_requests[0]["model"] == "qwen3:8b"
        proc = assistant._proc
        assert isinstance(proc, FakeProc)
        assistant.shutdown()
        assert proc.terminated and assistant._proc is None
    finally:
        [s.close() for s in servers]


def test_shutdown_leaves_an_ollama_it_did_not_start_alone(running):
    assistant.bootstrap()
    assert assistant._proc is None
    assistant.shutdown()                                   # must not raise


def test_reports_when_ollama_is_not_installed(monkeypatch):
    stopped(monkeypatch)
    monkeypatch.setattr(assistant, "find_ollama_executable", lambda: None)
    assistant.bootstrap()
    st = assistant.status()
    assert assistant._state["phase"] == "missing" and "ollama.com" in st["error"] and st["starting"] is False and st["available"] is False


def test_can_be_switched_off_but_the_button_still_works(monkeypatch):
    stopped(monkeypatch)
    monkeypatch.setenv("EMOTISENSE_AUTOSTART_OLLAMA", "0")
    monkeypatch.setattr(assistant, "_spawn", lambda exe: pytest.fail("auto-start is off"))
    assistant.bootstrap()
    assert assistant._state["phase"] == "off" and "ollama serve" in assistant.status()["error"]
    monkeypatch.setattr(assistant, "find_ollama_executable", lambda: None)
    assistant.bootstrap(force=True)                        # the "Check again" button forces an attempt
    assert assistant._state["phase"] == "missing"


def test_reports_a_start_that_never_answers(monkeypatch):
    stopped(monkeypatch)
    monkeypatch.setenv("EMOTISENSE_OLLAMA_WAIT", "1")
    monkeypatch.setattr(assistant, "find_ollama_executable", lambda: "ollama")
    monkeypatch.setattr(assistant, "_spawn", lambda exe: FakeProc())
    assistant.bootstrap()
    assert assistant._state["phase"] == "failed" and "did not answer" in assistant._state["message"]


def test_reports_when_the_process_cannot_be_launched(monkeypatch):
    stopped(monkeypatch)
    monkeypatch.setattr(assistant, "find_ollama_executable", lambda: "ollama")
    def boom(exe): raise OSError("access denied")
    monkeypatch.setattr(assistant, "_spawn", boom)
    assistant.bootstrap()
    assert assistant._state["phase"] == "failed" and "access denied" in assistant._state["message"]


def test_warm_up_is_skipped_without_models_and_never_blocks_readiness(running):
    running.models = []
    assistant.bootstrap()
    assert assistant._state["phase"] == "ready" and running.generate_requests == []
    running.models = ["qwen3:8b"]
    def broken(timeout=240.0): raise RuntimeError("out of memory")
    import pytest as _p
    mp = _p.MonkeyPatch(); mp.setattr(assistant, "warm_up", broken)
    try:
        assistant.bootstrap()
    finally:
        mp.undo()
    assert assistant._state["phase"] == "ready" and "warm-up skipped" in assistant._state["message"]


def test_status_tells_the_ui_that_startup_is_in_progress(monkeypatch):
    stopped(monkeypatch)
    assistant._set("starting", "Starting Ollama…")
    j = TestClient(server.app).get("/api/assistant/status").json()
    assert j["starting"] is True and j["available"] is False and j["phase"] == "starting"
    assistant._set("warming", "Loading qwen3:8b into memory…")
    assert TestClient(server.app).get("/api/assistant/status").json()["warming"] is True


def test_background_start_is_not_run_twice(running):
    assistant._set("starting", "x")
    assert assistant.bootstrap_async() is False
    assistant._set("idle", "")
    assert assistant.bootstrap_async() is True
    deadline = time.time() + 5
    while assistant._state["phase"] != "ready" and time.time() < deadline:
        time.sleep(0.05)
    assert assistant._state["phase"] == "ready"


def test_start_endpoint_returns_immediately_and_reports_phase(running):
    r = TestClient(server.app).post("/api/assistant/start").json()
    assert r["started"] is True and r["phase"] in ("checking", "starting", "warming", "ready")
    deadline = time.time() + 5
    while assistant._state["phase"] != "ready" and time.time() < deadline:
        time.sleep(0.05)
    again = TestClient(server.app).post("/api/assistant/start").json()
    assert again["started"] is True and again["available"] is True


def test_server_startup_triggers_the_background_start(running):
    with TestClient(server.app) as c:                       # entering the context runs the lifespan (server start)
        deadline = time.time() + 5
        while assistant._state["phase"] != "ready" and time.time() < deadline:
            time.sleep(0.05)
        assert assistant._state["phase"] == "ready"
        assert c.get("/api/assistant/status").json()["available"] is True
    assert running.generate_requests, "the model should have been loaded at start-up"


def test_finds_the_windows_install_even_when_it_is_not_on_path(monkeypatch, tmp_path):
    exe = tmp_path / "Programs" / "Ollama" / "ollama.exe"
    exe.parent.mkdir(parents=True); exe.write_text("")
    monkeypatch.setattr(assistant.shutil, "which", lambda name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert assistant.find_ollama_executable() == str(exe)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "nowhere"))
    monkeypatch.delenv("ProgramFiles", raising=False)
    if not any(Path(p).is_file() for p in ("/usr/local/bin/ollama", "/usr/bin/ollama", "/opt/homebrew/bin/ollama")):
        assert assistant.find_ollama_executable() is None
    monkeypatch.setattr(assistant.shutil, "which", lambda name: "/found/ollama")
    assert assistant.find_ollama_executable() == "/found/ollama"
