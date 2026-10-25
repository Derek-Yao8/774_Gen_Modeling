"""Independent ChatGPT-authenticated CLI adapter, adapted from Project 1.
No API-key fallback; no dataset arrays sent to the planner.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from .planning import SCHEMA as RESPONSE_SCHEMA

class PlannerError(RuntimeError):
    pass

def codex_environment():
    # Do not inspect, copy or print credentials. Let Codex use its stored login.
    blocked = {"OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE",
               "CODEX_ACCESS_TOKEN", "CODEX_AUTH_JSON", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"}
    return {k: v for k, v in os.environ.items() if k.upper() not in blocked}


class CodexPlanner:
    backend = 'codex_cli_chatgpt'
    def __init__(self, executable=None, timeout=180):
        self.executable = executable or shutil.which("codex")
        if not self.executable:
            raise PlannerError("Codex CLI not found. Install Codex and sign in with ChatGPT; no API fallback is supported.")
        self.timeout = timeout

    def _base(self):
        return [self.executable, "-c", 'forced_login_method="chatgpt"',
                "-c", 'model_provider="openai"']

    def check_auth(self):
        result = subprocess.run(self._base() + ["login", "status"], env=codex_environment(),
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        status = result.stdout + result.stderr
        if result.returncode or "Logged in using ChatGPT" not in status:
            raise PlannerError("ChatGPT-authenticated Codex is required. Run codex login. API-key login is not accepted.")

    def propose(self, prompt, directory, *, images=()):
        images = [Path(path).resolve() for path in images]
        for path in images:
            if not path.is_file():
                raise PlannerError('Image attachment is missing: '+str(path))
        self.check_auth()
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        schema = directory / "response_schema.json"
        output = directory / "response.json"
        schema.write_text(json.dumps(RESPONSE_SCHEMA), encoding="utf-8")
        (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
        command = self._base() + ["exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
            "-c", 'approval_policy="never"', "-c", 'web_search="disabled"',
            "-c", "project_doc_max_bytes=0", "--json", "--output-schema", str(schema),
            "--output-last-message", str(output), "--cd", str(directory)]
        for feature in ("shell_tool", "unified_exec", "apps", "browser_use", "computer_use", "multi_agent", "hooks", "image_generation"):
            command += ["--disable", feature]
        attachments = [{'image_index':i+1, 'path':str(path),
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                       for i,path in enumerate(images)]
        (directory / 'image_attachments.json').write_text(json.dumps(attachments, indent=2), encoding='utf-8')
        for path in images:
            command += ['--image='+str(path)]
        command += ['--', '-']
        try:
            result = subprocess.run(command, input=prompt, env=codex_environment(), capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise PlannerError("Codex timed out; no retry or paid fallback was attempted.") from exc
        (directory / "events.jsonl").write_text(result.stdout, encoding="utf-8")
        (directory / "stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode or not output.exists():
            raise PlannerError(f"Codex failed (exit {result.returncode}). See {directory / 'stderr.txt'}. No paid fallback.")
        return json.loads(output.read_text(encoding="utf-8"))

