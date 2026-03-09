"""
Azure DevOps Pull Request Helper.

Creates branches, commits changes, and opens pull requests
against the Azure DevOps repo using the REST API.

Used by the Knowledge Updater Agent to propose doc corrections.
"""

import base64
import json
import logging
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from brain_ai.config import get_config

log = logging.getLogger(__name__)


class AzureDevOpsPR:
    """Helper to create branches and pull requests in Azure DevOps."""

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            cfg = get_config()

        ado = cfg["azure_devops"]
        self.pat = ado["pat"]
        self.repo_url = ado["repo_url"]  # e.g. https://msazure.visualstudio.com/One/_git/Mgmt-RecoverySvcs-BackupMgmt
        self.target_branch = ado.get("branch", "feature/add-agent-kt-memory-doc")

        # Parse org, project, repo from the URL
        self._parse_repo_url()

        # Base64 PAT for REST API auth
        self._auth_header = "Basic " + base64.b64encode(
            f":{self.pat}".encode("utf-8")
        ).decode("utf-8")

        self.clone_dir = Path(cfg["paths"]["repo_clone_dir"]).resolve()

    def _parse_repo_url(self):
        """Extract org, project, and repo name from Azure DevOps URL."""
        # Format: https://dev.azure.com/{org}/{project}/_git/{repo}
        # or:     https://{org}.visualstudio.com/{project}/_git/{repo}
        url = self.repo_url.rstrip("/")
        parts = url.split("/")

        git_idx = None
        for i, p in enumerate(parts):
            if p == "_git":
                git_idx = i
                break

        if git_idx is None:
            raise ValueError(f"Cannot parse Azure DevOps repo URL: {self.repo_url}")

        self.repo_name = parts[git_idx + 1]
        self.project = parts[git_idx - 1]

        # Determine org base URL
        if "visualstudio.com" in url:
            # https://{org}.visualstudio.com/{project}/_git/{repo}
            host_part = url.split("://")[1].split("/")[0]  # {org}.visualstudio.com
            self.api_base = f"https://{host_part}/{self.project}/_apis"
        elif "dev.azure.com" in url:
            # https://dev.azure.com/{org}/{project}/_git/{repo}
            org = parts[3]
            self.api_base = f"https://dev.azure.com/{org}/{self.project}/_apis"
        else:
            raise ValueError(f"Unsupported Azure DevOps URL format: {self.repo_url}")

        log.info(
            "Azure DevOps: project=%s, repo=%s, api_base=%s",
            self.project, self.repo_name, self.api_base,
        )

    def _api_request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        api_version: str = "7.1",
    ) -> Dict[str, Any]:
        """Make an authenticated Azure DevOps REST API call."""
        url = f"{self.api_base}/{path}"
        separator = "&" if "?" in url else "?"
        url += f"{separator}api-version={api_version}"

        data = json.dumps(body).encode("utf-8") if body else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", self._auth_header)
        req.add_header("Content-Type", "application/json")

        log.debug("API %s %s", method, url)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_body = resp.read().decode("utf-8")
                return json.loads(resp_body) if resp_body else {}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            log.error("API error %d: %s\n%s", e.code, url, error_body[:500])
            raise RuntimeError(f"Azure DevOps API error {e.code}: {error_body[:300]}")

    def _get_branch_ref(self, branch_name: str) -> Optional[str]:
        """Get the object ID (SHA) of a branch. Returns None if not found."""
        path = f"git/repositories/{self.repo_name}/refs?filter=heads/{branch_name}"
        result = self._api_request("GET", path)
        refs = result.get("value", [])
        if refs:
            return refs[0]["objectId"]
        return None

    def _create_branch(self, new_branch: str, source_branch: str) -> str:
        """
        Create a new branch from source_branch.
        Returns the object ID of the new branch head.
        """
        source_sha = self._get_branch_ref(source_branch)
        if not source_sha:
            raise RuntimeError(f"Source branch '{source_branch}' not found.")

        body = [
            {
                "name": f"refs/heads/{new_branch}",
                "oldObjectId": "0000000000000000000000000000000000000000",
                "newObjectId": source_sha,
            }
        ]

        path = f"git/repositories/{self.repo_name}/refs"
        result = self._api_request("POST", path, body)

        created = result.get("value", [])
        if not created or not created[0].get("success", True):
            # Branch might already exist — get its SHA
            existing_sha = self._get_branch_ref(new_branch)
            if existing_sha:
                log.info("Branch '%s' already exists (SHA: %s).", new_branch, existing_sha[:8])
                return existing_sha
            raise RuntimeError(f"Failed to create branch '{new_branch}': {result}")

        log.info("Created branch '%s' from '%s'.", new_branch, source_branch)
        return source_sha

    def push_file_change(
        self,
        branch: str,
        file_path: str,
        new_content: str,
        commit_message: str,
    ) -> Dict[str, Any]:
        """
        Push a single file change (edit) to a branch.

        Args:
            branch: Branch name (without refs/heads/)
            file_path: Path within the repo (e.g. "docs/agentKT/docs/agentKT/Feature_X.md")
            new_content: The full new file content
            commit_message: Commit message

        Returns:
            The push result dict.
        """
        # Get latest commit on the branch
        branch_sha = self._get_branch_ref(branch)
        if not branch_sha:
            raise RuntimeError(f"Branch '{branch}' not found.")

        body = {
            "refUpdates": [
                {
                    "name": f"refs/heads/{branch}",
                    "oldObjectId": branch_sha,
                }
            ],
            "commits": [
                {
                    "comment": commit_message,
                    "changes": [
                        {
                            "changeType": "edit",
                            "item": {"path": f"/{file_path}"},
                            "newContent": {
                                "content": new_content,
                                "contentType": "rawtext",
                            },
                        }
                    ],
                }
            ],
        }

        path = f"git/repositories/{self.repo_name}/pushes"
        result = self._api_request("POST", path, body)
        log.info("Pushed commit to '%s': %s", branch, commit_message)
        return result

    def create_pull_request(
        self,
        source_branch: str,
        target_branch: str,
        title: str,
        description: str,
    ) -> Dict[str, Any]:
        """
        Create a pull request from source_branch into target_branch.

        Returns the PR details including the PR URL.
        """
        body = {
            "sourceRefName": f"refs/heads/{source_branch}",
            "targetRefName": f"refs/heads/{target_branch}",
            "title": title,
            "description": description,
        }

        path = f"git/repositories/{self.repo_name}/pullrequests"
        result = self._api_request("POST", path, body)

        pr_id = result.get("pullRequestId")
        pr_url = result.get("url", "")

        # Build a web URL for the user
        web_url = (
            f"{self.repo_url}/pullrequest/{pr_id}"
            if pr_id
            else pr_url
        )

        log.info("Created PR #%s: %s", pr_id, title)
        return {
            "pr_id": pr_id,
            "title": title,
            "web_url": web_url,
            "status": result.get("status", "active"),
            "raw": result,
        }

    def create_correction_pr(
        self,
        file_path: str,
        new_content: str,
        correction_summary: str,
    ) -> Dict[str, Any]:
        """
        End-to-end: create a branch, push the corrected file, and open a PR.

        Args:
            file_path: Path within the repo (e.g. "docs/agentKT/docs/agentKT/Feature_X.md")
            new_content: The full corrected file content
            correction_summary: Short description of the correction

        Returns:
            Dict with pr_id, web_url, branch_name, etc.
        """
        # Generate a unique branch name
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe_summary = re.sub(r'[^a-z0-9_-]', '-', correction_summary[:40].lower()).strip('-')
        safe_summary = re.sub(r'-{2,}', '-', safe_summary)  # collapse multiple dashes
        branch_name = f"BCDR-devai/doc-correction/{safe_summary}-{timestamp}"

        log.info("Creating correction PR: branch=%s, file=%s", branch_name, file_path)

        # 1. Create branch from target
        self._create_branch(branch_name, self.target_branch)

        # 2. Push the corrected file
        commit_msg = f"[BCDR DeveloperAI] Doc correction: {correction_summary}"
        self.push_file_change(branch_name, file_path, new_content, commit_msg)

        # 3. Create PR
        pr_result = self.create_pull_request(
            source_branch=branch_name,
            target_branch=self.target_branch,
            title=f"[BCDR DeveloperAI Auto-Correction] {correction_summary}",
            description=(
                f"## Automated Documentation Correction\n\n"
                f"**File:** `{file_path}`\n\n"
                f"**Summary:** {correction_summary}\n\n"
                f"---\n"
                f"_This PR was created automatically by the BCDR DeveloperAI Knowledge Updater Agent "
                f"based on a user correction during a chat session._"
            ),
        )

        pr_result["branch_name"] = branch_name
        pr_result["file_path"] = file_path
        return pr_result

    def create_batch_correction_pr(
        self,
        file_changes: List[Dict[str, str]],
        overall_summary: str,
    ) -> Dict[str, Any]:
        """
        End-to-end: create a branch, push multiple file changes in a single
        commit, and open one PR for all corrections in the session.

        Args:
            file_changes: List of {"file_path": "...", "new_content": "...", "summary": "..."}
            overall_summary: Short description covering all corrections

        Returns:
            Dict with pr_id, web_url, branch_name, files_changed, etc.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe_summary = re.sub(r'[^a-z0-9_-]', '-', overall_summary[:40].lower()).strip('-')
        safe_summary = re.sub(r'-{2,}', '-', safe_summary)  # collapse multiple dashes
        branch_name = f"BCDR-devai/doc-corrections/{safe_summary}-{timestamp}"

        log.info(
            "Creating batch correction PR: branch=%s, files=%d",
            branch_name, len(file_changes),
        )

        # 1. Create branch from target
        self._create_branch(branch_name, self.target_branch)

        # 2. Push all file changes in a single commit
        branch_sha = self._get_branch_ref(branch_name)
        if not branch_sha:
            raise RuntimeError(f"Branch '{branch_name}' not found after creation.")

        changes = []
        for fc in file_changes:
            # Support explicit changeType ("add" for new files, "edit" for updates)
            change_type = fc.get("changeType", "edit")
            changes.append({
                "changeType": change_type,
                "item": {"path": f"/{fc['file_path']}"},
                "newContent": {
                    "content": fc["new_content"],
                    "contentType": "rawtext",
                },
            })

        body = {
            "refUpdates": [
                {
                    "name": f"refs/heads/{branch_name}",
                    "oldObjectId": branch_sha,
                }
            ],
            "commits": [
                {
                    "comment": f"[BCDR DeveloperAI] Doc corrections: {overall_summary}",
                    "changes": changes,
                }
            ],
        }

        path = f"git/repositories/{self.repo_name}/pushes"
        self._api_request("POST", path, body)
        log.info("Pushed %d file changes to '%s'.", len(file_changes), branch_name)

        # 3. Build PR description with per-file details
        file_list = "\n".join(
            f"- `{fc['file_path']}` — {fc.get('summary', 'updated')} "
            f"({'new' if fc.get('changeType') == 'add' else 'updated'})"
            for fc in file_changes
        )
        description = (
            f"## Automated Documentation Corrections\n\n"
            f"**{len(file_changes)} file(s) updated:**\n{file_list}\n\n"
            f"**Summary:** {overall_summary}\n\n"
            f"---\n"
            f"_This PR was created automatically by the BCDR DeveloperAI Knowledge Updater Agent "
            f"based on user corrections during a chat session._"
        )

        # 4. Create PR
        pr_result = self.create_pull_request(
            source_branch=branch_name,
            target_branch=self.target_branch,
            title=f"[BCDR DeveloperAI] {overall_summary}",
            description=description,
        )

        pr_result["branch_name"] = branch_name
        pr_result["files_changed"] = len(file_changes)
        return pr_result
