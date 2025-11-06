import json
from importlib import resources

from flask.testing import FlaskClient


def smoke_test_dashboard() -> None:
    from merlion.dashboard.server import server

    with server.test_client() as client:  # type: FlaskClient
        root = client.get("/")
        assert root.status_code == 200, f"Unexpected status for /: {root.status_code}"
        if b"Merlion Dashboard" not in root.data:
            raise AssertionError("Dashboard HTML did not contain expected title snippet")

        layout = client.get("/_dash-layout")
        assert layout.status_code == 200, "Dash layout endpoint failed"
        json.loads(layout.data.decode("utf-8"))

        deps = client.get("/_dash-dependencies")
        assert deps.status_code == 200, "Dash dependencies endpoint failed"

        assets_dir = resources.files("merlion.dashboard") / "assets"
        assert assets_dir.is_dir(), f"Dashboard assets are missing at {assets_dir}"


if __name__ == "__main__":
    smoke_test_dashboard()
    print("Merlion dashboard smoke test passed.")
