def test_health_contract(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    for k in ("status", "api_version", "exploration_model", "production_models", "earth_engine", "cache"):
        assert k in d
    assert d["status"] == "ok"
    assert d["exploration_model"] and d["production_models"] and d["cache"]


def test_health_exposes_no_secrets(client):
    text = client.get("/api/health").text.lower()
    for bad in ("client_email", "private_key", "/etc/secrets", "gee_key", "service_account", "credential"):
        assert bad not in text


def test_debug_endpoint_removed(client):
    assert client.get("/debug_ee").status_code == 404


def test_repository_files_not_served(client):
    for path in ("/main.py", "/.git/config", "/models/model_manifest.json", "/data/production_history.csv",
                 "/requirements.txt", "/config/risk_policy.json"):
        r = client.get(path)
        assert r.status_code == 404, path
        assert r.json()["error"] == "NOT_FOUND"


def test_frontend_files_served(client):
    for path, ctype in (("/", "text/html"), ("/index.html", "text/html"), ("/app.js", "javascript"), ("/style.css", "text/css")):
        r = client.get(path)
        assert r.status_code == 200 and ctype in r.headers["content-type"]


def test_no_cors_by_default(client):
    r = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_unknown_route_structured_404(client):
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404 and r.json()["error"] == "NOT_FOUND"


def test_legacy_reserve_endpoint_removed(client):
    assert client.get("/reserve_grid").status_code == 404
