"""ROUND 4.2: dry-run --help and entry-point tests."""
import subprocess, sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = str(_PROJECT_ROOT / "scripts" / "run_local_avoidance_dry_run.py")


class TestDryRunHelp:
    def test_help_exit_code_zero(self):
        result = subprocess.run(
            [sys.executable, SCRIPT, "--help"],
            capture_output=True, text=True, timeout=10,
            env={**__import__("os").environ,
                 "PYTHONDONTWRITEBYTECODE": "1",
                 "AIRSIM_PYTHONCLIENT_PATH": "",
                 "PYTHONPATH": ""},
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "--settings-json" in result.stdout
        assert "--frames" in result.stdout
        assert "--output" in result.stdout

    def test_missing_settings_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, SCRIPT, "--frames", "1"],
            capture_output=True, text=True, timeout=10,
            env={**__import__("os").environ,
                 "PYTHONDONTWRITEBYTECODE": "1",
                 "AIRSIM_PYTHONCLIENT_PATH": "",
                 "PYTHONPATH": ""},
        )
        assert result.returncode != 0, "--settings-json required but not provided"

    def test_no_main_typo(self):
        """Verify the entry point is 'main()', not '__main__()'."""
        content = Path(SCRIPT).read_text(encoding="utf-8")
        assert "if __name__ == \"__main__\":" in content
        assert "main()" in content
        assert "__main__()" not in content
