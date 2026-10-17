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


def post(c, files=None, headers=None, **extra):
    f = files or demo_files()
    form = {"task_yaml": f["task_yaml"], "models_yaml": f["models_yaml"], "n_boot": "200", "concurrency": "1", **extra}
    return c.post("/api/runs", data=form, headers=headers,
                  files={"data": ("data.csv", f["data_csv"].encode(), "text/csv")})


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


@pytest.mark.parametrize("bad", ["run-1", "run-20250101-000000%0A", "run-20250101-000000-", "RUN-20250101-000000"])
def test_run_id_is_checked_by_the_handler_before_any_filesystem_access(tmp_path, monkeypatch, bad):
    # These ids contain no slash, so the router passes them to the handler; the regex must stop them there.
    c = client(tmp_path)

    def boom(*a, **k):
        raise AssertionError("filesystem touched before the id was validated")

    monkeypatch.setattr(webapp.Path, "is_dir", boom)
    seen = []
    real = webapp.RUN_ID.fullmatch
    monkeypatch.setattr(webapp, "RUN_ID", type("R", (), {"fullmatch": staticmethod(lambda x: seen.append(x) or real(x))}))
    assert c.get(f"/api/runs/{bad}").status_code == 404
    assert seen == [bad.replace("%0A", "\n")]  # the handler itself ran the check and rejected it


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


def test_failure_is_one_line_and_the_key_value_never_appears_in_any_response(tmp_path, monkeypatch):
    secret = "sk-test-SECRET-1234567890abcdef"
    monkeypatch.setenv("MY_PROVIDER_KEY", secret)
    f = demo_files()
    f["models_yaml"] = ("- {name: m, kind: llm, provider: mock, model: m, api_key_env: MY_PROVIDER_KEY, "
                        "params: {accuracy: 0.9, cost_per_call: 0.01}}\n")

    def bad(*a, **k):
        raise RuntimeError(f"auth failed for {secret}\n  File \"x.py\", line 3\nTraceback (most recent call last)")

    monkeypatch.setattr(webapp, "run_experiment", bad)
    c = client(tmp_path)
    rid = post(c, f).json()["id"]
    s = wait_done(c, rid)
    assert s["state"] == "failed" and "\n" not in s["error"] and "[redacted]" in s["error"]
    bodies = [c.get(p).text for p in ("/api/demo", "/api/runs", f"/api/runs/{rid}", f"/api/runs/{rid}/report")]
    assert all(secret not in b for b in bodies)


def test_secret_in_a_config_error_is_redacted(tmp_path, monkeypatch):
    secret = "sk-test-SECRET-1234567890abcdef"
    monkeypatch.setenv("SOME_API_KEY", secret)

    def leaky(path):
        raise ValueError(f"bad value {secret} in {path}")

    monkeypatch.setattr(webapp, "load_task", leaky)
    r = post(client(tmp_path))
    assert r.status_code == 400 and secret not in r.text and "[redacted]" in r.json()["error"]


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


def test_security_headers_are_set(tmp_path):
    r = client(tmp_path).get("/api/runs")
    c = client(tmp_path)
    for resp in (r, c.get("/api/runs/run-1"), c.get("/"), c.post("/api/runs", headers={"origin": "https://evil.example"})):
        assert resp.headers["x-frame-options"] == "SAMEORIGIN"  # the app's own report iframe must work
        assert resp.headers["content-security-policy"] == "frame-ancestors 'self'"
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert "DENY" not in str(resp.headers)


# ---- cross-site defence -------------------------------------------------------------------------------------

def _lock_is_free_and_nothing_left(c, tmp_path):
    assert not list((tmp_path / "runs").glob("run-*"))
    ok = post(c)  # a normal run starts, so the lock was not taken
    assert ok.status_code == 200
    wait_done(c, ok.json()["id"])


@pytest.mark.parametrize("origin", ["https://evil.example", "http://evil.example:8765", "null",
                                    "http://127.0.0.1", "http://127.0.0.1:9999", "http://localhost:8765",
                                    "https://127.0.0.1:8765"])
def test_post_with_a_foreign_origin_is_403_before_anything_happens(tmp_path, origin):
    c = client(tmp_path)
    r = post(c, headers={"origin": origin})
    assert r.status_code == 403 and "\n" not in r.json()["error"].strip()
    _lock_is_free_and_nothing_left(c, tmp_path)


def test_post_with_cross_site_fetch_metadata_is_403(tmp_path):
    c = client(tmp_path)
    for site in ("cross-site", "same-site"):
        assert post(c, headers={"sec-fetch-site": site}).status_code == 403
    _lock_is_free_and_nothing_left(c, tmp_path)


def test_same_origin_none_fetch_site_and_no_origin_posts_work(tmp_path):
    c = client(tmp_path)
    r = post(c, headers={"origin": "http://127.0.0.1:8765", "sec-fetch-site": "same-origin"})
    assert r.status_code == 200
    wait_done(c, r.json()["id"])
    r = post(c, headers={"origin": "HTTP://127.0.0.1:8765", "sec-fetch-site": "none"})
    assert r.status_code == 200
    wait_done(c, r.json()["id"])


def test_get_with_a_foreign_origin_still_works(tmp_path):
    c = client(tmp_path)
    assert c.get("/api/runs", headers={"origin": "https://evil.example", "sec-fetch-site": "cross-site"}).status_code == 200


# ---- lock-leak paths and input limits -----------------------------------------------------------------------

def test_new_run_dir_failure_releases_the_lock(tmp_path, monkeypatch):
    c = client(tmp_path)
    real = webapp.new_run_dir

    def broken(*a):
        raise OSError("disk on fire")

    monkeypatch.setattr(webapp, "new_run_dir", broken)
    r = post(c)
    assert r.status_code == 400 and "disk on fire" in r.json()["error"]
    monkeypatch.setattr(webapp, "new_run_dir", real)
    ok = post(c)
    assert ok.status_code == 200
    wait_done(c, ok.json()["id"])


def test_thread_start_failure_releases_the_lock_and_marks_the_run_failed(tmp_path, monkeypatch):
    c = client(tmp_path)
    real_start = webapp.threading.Thread.start

    def start(self):
        if getattr(self._target, "__name__", "") == "work":
            raise RuntimeError("can't start new thread")
        return real_start(self)

    monkeypatch.setattr(webapp.threading.Thread, "start", start)
    r = post(c)
    assert r.status_code == 500 and "\n" not in r.json()["error"].strip()
    listed = c.get("/api/runs").json()
    assert len(listed) == 1
    st = c.get(f"/api/runs/{listed[0]['id']}").json()
    assert st["state"] == "failed" and "thread" in st["error"]
    monkeypatch.setattr(webapp.threading.Thread, "start", real_start)
    ok = post(c)  # not 409: the lock was released
    assert ok.status_code == 200
    wait_done(c, ok.json()["id"])


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")  # SystemExit is deliberate
def test_base_exception_in_the_worker_does_not_leave_the_run_running(tmp_path, monkeypatch):
    real = webapp.run_experiment

    def die(*a, **k):
        raise SystemExit(3)

    monkeypatch.setattr(webapp, "run_experiment", die)
    c = client(tmp_path)
    first = post(c)
    s = wait_done(c, first.json()["id"])
    assert s["state"] == "failed" and s["error"]
    monkeypatch.setattr(webapp, "run_experiment", real)
    ok = post(c)
    assert ok.status_code == 200
    wait_done(c, ok.json()["id"])


@pytest.mark.parametrize("extra", [{"max_cost": "-1"}, {"max_cost": "nan"}, {"max_cost": "inf"},
                                   {"n_boot": "1000001"}])
def test_numeric_limits_are_422(tmp_path, extra):
    c = client(tmp_path)
    assert post(c, **extra).status_code == 422
    assert not list((tmp_path / "runs").glob("run-*"))


@pytest.mark.parametrize("field", ["task_yaml", "models_yaml", "pricing_yaml"])
def test_oversized_yaml_is_a_one_line_400_error(tmp_path, field):
    c = client(tmp_path)
    f = demo_files()
    big = "# " + "x" * 70_000 + "\n"
    if field == "pricing_yaml":
        r = post(c, f, pricing_yaml=big)
    else:
        f[field] = f[field] + big
        r = post(c, f)
    assert r.status_code == 400 and set(r.json()) == {"error"} and "\n" not in r.json()["error"].strip()
    assert not list((tmp_path / "runs").glob("run-*"))


def test_corrupt_results_json_reads_as_not_done(tmp_path):
    c = client(tmp_path)
    d = tmp_path / "runs" / "run-20250101-000000"
    d.mkdir(parents=True)
    for payload in (b'{"recommendation": ', b"\xff\xfe\x00garbage", b""):
        (d / "results.json").write_bytes(payload)
        st = c.get("/api/runs/run-20250101-000000")
        assert st.status_code == 200 and st.json()["state"] == "unknown"
        assert c.get("/api/runs/run-20250101-000000/report").status_code == 404
        assert c.get("/api/runs").status_code == 200


import re  # noqa: E402


def test_index_is_one_offline_page_with_the_run_form(tmp_path):
    html = client(tmp_path).get("/").text
    assert 'id="run-form"' in html and 'id="demo-btn"' in html and 'id="report-frame"' in html
    assert not re.search(r'(src|href)="https?://', html) and "cdn" not in html.lower()
    assert 'sandbox' in html            # the report iframe is sandboxed
    assert "allow-scripts" not in html and "allow-same-origin" not in html
    # the page never asks for a key: no key/password input (the trust note may mention the api_key_env setting)
    assert "password" not in html.lower() and not re.search(r'name="[^"]*(key|token|secret)', html, re.I)
    assert "Only paste a models.yaml you wrote or trust" in html
    assert "innerhtml" not in html.lower()  # server strings go in through textContent only
    assert not re.search(r"outerHTML|insertAdjacentHTML|document\.write|eval\(", html)
    assert 'href="/static/favicon.svg"' in html and 'src="/static/symbol.svg"' in html


def test_static_serves_the_brand_files_and_no_traversal(tmp_path):
    c = client(tmp_path)
    for name in ("favicon.svg", "symbol.svg"):
        r = c.get(f"/static/{name}")
        assert r.status_code == 200 and r.headers["content-type"].startswith("image/svg+xml")
    for path in ("/static/../app.py", "/static/%2e%2e/app.py", "/static/..%2fapp.py", "/static/%2e%2e%2fapp.py",
                 "/static/index.html/../../app.py"):
        r = c.get(path)
        assert r.status_code in (400, 404) and "create_app" not in r.text, path
    # still behind the Host check
    assert c.get("/static/favicon.svg", headers={"host": "evil.example"}).status_code == 403


def _upload(c, name, content, f=None):
    f = f or demo_files()
    form = {"task_yaml": f["task_yaml"], "models_yaml": f["models_yaml"], "n_boot": "200", "concurrency": "1"}
    return c.post("/api/runs", data=form, files={"data": (name, content, "application/octet-stream")})


def test_jsonl_upload_runs_to_done(tmp_path):
    import pandas as pd
    c = client(tmp_path)
    csv_text = demo_files()["data_csv"]
    rows = pd.read_csv(__import__("io").StringIO(csv_text)).head(60)
    r = _upload(c, "data.JSONL", rows.to_json(orient="records", lines=True).encode())
    assert r.status_code == 200, r.text
    assert wait_done(c, r.json()["id"])["state"] == "done"


def test_unsupported_upload_type_is_a_one_line_400_and_leaves_no_folder(tmp_path):
    c = client(tmp_path)
    for name in ("data.txt", "", "data"):
        r = _upload(c, name, b"a,b" + bytes([10]) + b"1,2")
        assert r.status_code in (400, 422), (name, r.status_code)
        if r.status_code == 400:
            assert chr(10) not in r.json()["error"].strip() and ".csv" in r.json()["error"]
    assert not list((tmp_path / "runs").glob("run-*"))


def test_report_route_is_script_inert_and_other_routes_are_not_sandboxed(tmp_path):
    c = client(tmp_path)
    run_id = post(c).json()["id"]
    wait_done(c, run_id)
    rep = c.get(f"/api/runs/{run_id}/report")
    assert rep.headers["content-security-policy"] == "sandbox; frame-ancestors 'self'"
    assert c.get("/").headers["content-security-policy"] == "frame-ancestors 'self'"
    assert c.get(f"/api/runs/{run_id}").headers["content-security-policy"] == "frame-ancestors 'self'"


def test_error_text_never_contains_the_runs_folder_path(tmp_path):
    c = client(tmp_path)
    f = demo_files()
    f["models_yaml"] = "not: [a valid list"
    err = post(c, f).json()["error"]
    assert str(tmp_path) not in err and str((tmp_path / "runs").resolve()) not in err
    assert str(tmp_path).replace("\\", "/") not in err
