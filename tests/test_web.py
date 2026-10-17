import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

from tiebreak.demo import demo_files  # noqa: E402
from tiebreak.web import app as webapp  # noqa: E402


def client(tmp_path, **kw):
    return TestClient(webapp.create_app(tmp_path / "runs", **kw), base_url="http://127.0.0.1:8765")


def post(c, files=None, **extra):
    f = files or demo_files()
    form = {"task_yaml": f["task_yaml"], "models_yaml": f["models_yaml"], "n_boot": "200", "concurrency": "1", **extra}
    return c.post("/api/runs", data=form, files={"data": ("data.csv", f["data_csv"].encode(), "text/csv")})


def wait_done(c, run_id, seconds=90):
    end = time.time() + seconds
    while time.time() < end:
        s = c.get(f"/api/runs/{run_id}").json()
        if s["state"] != "running":
            return s
        time.sleep(0.3)
    raise AssertionError("run did not finish")


def test_demo_files_endpoint_and_empty_list(tmp_path):
    c = client(tmp_path)
    assert set(c.get("/api/demo").json()) == {"task_yaml", "models_yaml", "data_csv"}
    assert c.get("/api/runs").json() == []


def test_full_run_then_report(tmp_path):
    c = client(tmp_path)
    r = post(c)
    assert r.status_code == 200
    run_id = r.json()["id"]
    s = wait_done(c, run_id)
    assert s["state"] == "done" and s["recommendation"]["winner"]
    rep = c.get(f"/api/runs/{run_id}/report")
    assert rep.status_code == 200 and "SIMULATED" in rep.text and "<script" not in rep.text.lower()
    assert c.get("/api/runs").json()[0]["id"] == run_id


def test_hosted_candidate_needs_consent_and_leaves_no_folder(tmp_path):
    c = client(tmp_path)
    f = demo_files()
    f["models_yaml"] = "- {name: h, kind: llm, provider: anthropic, model: some-model}\n"
    r = post(c, f)
    assert r.status_code == 400 and "send" in r.json()["error"].lower()
    assert not list((tmp_path / "runs").glob("run-*"))


def test_bad_config_gives_a_one_line_400(tmp_path):
    c = client(tmp_path)
    f = demo_files()
    f["task_yaml"] = "type: classify\n"
    r = post(c, f)
    assert r.status_code == 400 and "\n" not in r.json()["error"].strip()
    assert not list((tmp_path / "runs").glob("run-*"))


def test_foreign_host_header_is_rejected(tmp_path):
    c = client(tmp_path)
    assert c.get("/api/runs", headers={"host": "evil.example"}).status_code == 403


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8765", "127.0.0.1.evil.example",
                                  "localhost.evil.example", "[::2]", "10.0.0.5:8765", "", "evil.example:80@localhost"])
def test_foreign_hosts_get_403_on_every_route(tmp_path, host):
    c = client(tmp_path)
    for path in ("/", "/api/demo", "/api/runs", "/api/runs/run-20250101-000000"):
        assert c.get(path, headers={"host": host}).status_code == 403, (host, path)
    assert c.post("/api/runs", headers={"host": host}).status_code == 403


@pytest.mark.parametrize("host", ["localhost", "localhost:8765", "127.0.0.1", "127.0.0.1:8765",
                                  "[::1]", "[::1]:8765", "LOCALHOST:8765"])
def test_local_hosts_are_accepted(tmp_path, host):
    c = client(tmp_path)
    assert c.get("/api/runs", headers={"host": host}).status_code == 200


@pytest.mark.parametrize("bad", ["run-1", "..", "run-20250101-000000/../x", "RUN-20250101-000000"])
def test_bad_run_ids_are_404(tmp_path, bad):
    c = client(tmp_path)
    assert c.get(f"/api/runs/{bad}").status_code == 404
    assert c.get(f"/api/runs/{bad}/report").status_code == 404


@pytest.mark.parametrize("bad", ["run-20250101-000000%2F..%2Fx", "run-20250101-000000%0A", "%2e%2e", "run-20250101-000000/report/x",
                                 "run-20250101-000000-", "run-2025010-000000", "Run-20250101-000000"])
def test_tricky_run_ids_are_404_and_never_touch_the_disk(tmp_path, bad, monkeypatch):
    c = client(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "run-20250101-000000").mkdir()  # a real run exists next to the probes
    assert c.get(f"/api/runs/{bad}").status_code == 404
    assert c.get(f"/api/runs/{bad}/report").status_code == 404


@pytest.mark.parametrize("bad", ["run-20250101-000000%2f", "run-20250101-000000/", "run-20250101-000099/"])
def test_trailing_slash_variants_only_ever_reach_a_valid_id(tmp_path, bad):
    # Starlette redirects a trailing slash to the slash-less route, where the id is validated again.
    c = client(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "run-20250101-000000").mkdir()
    r = c.get(f"/api/runs/{bad}")
    assert r.status_code == (200 if bad.startswith("run-20250101-000000") else 404)
    assert r.json().get("id", "run-20250101-000000") == "run-20250101-000000"


def test_run_id_is_checked_before_any_filesystem_access(tmp_path, monkeypatch):
    c = client(tmp_path)

    def boom(*a, **k):
        raise AssertionError("filesystem touched before the id was validated")

    monkeypatch.setattr(webapp.Path, "is_dir", boom)
    assert c.get("/api/runs/..").status_code == 404
    assert c.get("/api/runs/..%2f..%2fetc/report").status_code == 404


def test_listing_ignores_folders_that_are_not_runs(tmp_path):
    c = client(tmp_path)
    (tmp_path / "runs" / "run-20250101-000000").mkdir(parents=True)
    (tmp_path / "runs" / "run-evil").mkdir()
    (tmp_path / "runs" / "notes").mkdir()
    assert [r["id"] for r in c.get("/api/runs").json()] == ["run-20250101-000000"]


def test_upload_too_large_is_413(tmp_path):
    c = client(tmp_path, max_upload_mb=0.0001)
    assert post(c).status_code == 413


def test_bad_form_values_are_rejected_without_a_run(tmp_path):
    c = client(tmp_path)
    assert post(c, concurrency="99").status_code == 422
    assert post(c, n_boot="5").status_code == 422
    assert not list((tmp_path / "runs").glob("run-*"))


def test_failed_validation_releases_the_lock(tmp_path):
    c = client(tmp_path)
    f = demo_files()
    f["task_yaml"] = "type: classify\n"
    assert post(c, f).status_code == 400
    assert post(c, f).status_code == 400  # not 409: the lock was released
    ok = post(c)
    assert ok.status_code == 200
    wait_done(c, ok.json()["id"])


def test_failure_is_one_line_without_traceback_and_no_key_in_any_response(tmp_path, monkeypatch):
    secret = "sk-test-SECRET-1234567890"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    def bad(*a, **k):
        raise RuntimeError("line one\n  File \"x.py\", line 3\nTraceback (most recent call last)")

    monkeypatch.setattr(webapp, "run_experiment", bad)
    c = client(tmp_path)
    rid = post(c).json()["id"]
    s = wait_done(c, rid)
    assert s["state"] == "failed" and "\n" not in s["error"]
    bodies = [c.get(p).text for p in ("/api/demo", "/api/runs", f"/api/runs/{rid}", f"/api/runs/{rid}/report")]
    assert all(secret not in b for b in bodies)


def test_second_run_while_active_is_409_and_lock_is_released_after_failure(tmp_path, monkeypatch):
    gate = threading.Event()
    started = threading.Event()
    real = webapp.run_experiment

    def slow(*a, **k):
        started.set()
        gate.wait(10)
        raise RuntimeError("boom")

    monkeypatch.setattr(webapp, "run_experiment", slow)
    c = client(tmp_path)
    first = post(c)
    assert first.status_code == 200 and started.wait(5)
    assert post(c).status_code == 409
    gate.set()
    s = wait_done(c, first.json()["id"])
    assert s["state"] == "failed" and "boom" in s["error"]
    monkeypatch.setattr(webapp, "run_experiment", real)
    assert post(c).status_code == 200  # the lock was released
