"""Opt-in Docker build-context verification for the private publisher image."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(not os.getenv("MPF_TEST_DOCKER"), reason="set MPF_TEST_DOCKER=1 to build the local publisher container")
def test_forecast_publisher_image_imports_and_starts_from_docker_build_context() -> None:
    if not shutil.which("docker"):
        pytest.skip("docker is unavailable")
    if subprocess.run(["docker", "info"], capture_output=True).returncode:
        pytest.skip("docker daemon is unavailable")
    root = Path(__file__).resolve().parents[2]
    tag = "mpf-forecast-publisher-local-test"
    subprocess.run(["docker", "build", "-f", "agentic_vol_regime_app/Dockerfile.forecast-publisher", "-t", tag, "."], cwd=root, check=True)
    subprocess.run(["docker", "run", "--rm", "--entrypoint", "python", tag, "-c", "from agentic_vol_regime_app.forecast_publisher_service import create_app; assert create_app()"], cwd=root, check=True)
