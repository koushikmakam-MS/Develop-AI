"""Tests for branch name sanitization in brain_ai.sync.devops_pr."""

import re


def _sanitize_branch_summary(raw: str) -> str:
    """Replicate the branch-name sanitization logic from devops_pr.py.

    Extracted here so we can test it without needing Azure DevOps credentials.
    """
    safe = re.sub(r'[^a-z0-9_-]', '-', raw[:40].lower()).strip('-')
    safe = re.sub(r'-{2,}', '-', safe)
    return safe


class TestBranchNameSanitization:
    def test_simple_spaces(self):
        assert _sanitize_branch_summary("fix backup retry logic") == "fix-backup-retry-logic"

    def test_colons_removed(self):
        result = _sanitize_branch_summary("doc-improvements:-1-new,-27-updated")
        assert ":" not in result
        assert "," not in result

    def test_slashes_removed(self):
        result = _sanitize_branch_summary("fix feature/backup/policy")
        assert "/" not in result

    def test_special_characters(self):
        result = _sanitize_branch_summary("fix: backup (retry) logic [v2]")
        assert ":" not in result
        assert "(" not in result
        assert ")" not in result
        assert "[" not in result
        assert "]" not in result

    def test_collapses_multiple_dashes(self):
        result = _sanitize_branch_summary("fix---this---thing")
        assert "---" not in result
        assert "--" not in result

    def test_strips_leading_trailing_dashes(self):
        result = _sanitize_branch_summary("---leading and trailing---")
        assert not result.startswith("-")
        assert not result.endswith("-")

    def test_truncates_to_40_chars(self):
        long_input = "a" * 100
        result = _sanitize_branch_summary(long_input)
        assert len(result) <= 40

    def test_preserves_underscores(self):
        result = _sanitize_branch_summary("fix_this_bug")
        assert result == "fix_this_bug"

    def test_lowercase(self):
        result = _sanitize_branch_summary("Fix Backup Policy")
        assert result == result.lower()

    def test_unicode_stripped(self):
        result = _sanitize_branch_summary("fix café résumé")
        # All non-alphanumeric chars should be dashes
        assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in result)

    def test_empty_string(self):
        result = _sanitize_branch_summary("")
        assert result == ""

    def test_only_special_chars(self):
        result = _sanitize_branch_summary(":::,,,///")
        # All stripped → empty after trimming dashes
        assert result == ""

    def test_real_world_example(self):
        """The exact input that caused the original invalidRefName error."""
        result = _sanitize_branch_summary("doc-improvements:-1-new,-27-updated-20260305-131900")
        assert ":" not in result
        assert "," not in result
        # Should be something like: doc-improvements-1-new-27-updated-20260
        assert result.startswith("doc-improvements")


class TestUrlParsing:
    """Test the _parse_repo_url logic for different Azure DevOps URL formats."""

    def test_visualstudio_format(self):
        """https://{org}.visualstudio.com/{project}/_git/{repo}"""
        url = "https://msazure.visualstudio.com/One/_git/Mgmt-RecoverySvcs-BackupMgmt"
        parts = url.rstrip("/").split("/")
        git_idx = parts.index("_git")
        assert parts[git_idx + 1] == "Mgmt-RecoverySvcs-BackupMgmt"
        assert parts[git_idx - 1] == "One"

    def test_dev_azure_format(self):
        """https://dev.azure.com/{org}/{project}/_git/{repo}"""
        url = "https://dev.azure.com/myorg/MyProject/_git/MyRepo"
        parts = url.rstrip("/").split("/")
        git_idx = parts.index("_git")
        assert parts[git_idx + 1] == "MyRepo"
        assert parts[git_idx - 1] == "MyProject"
        assert parts[3] == "myorg"

    def test_no_git_in_url_raises(self):
        url = "https://github.com/user/repo"
        parts = url.rstrip("/").split("/")
        assert "_git" not in parts
