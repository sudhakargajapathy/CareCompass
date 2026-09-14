"""GitHub-native alerting: one issue per finding identity, closed on recovery.

Why issues and not the workflow's failure email alone: a scheduled
workflow's failure email goes to the LAST COMMITTER OF THE WORKFLOW FILE,
says only that the run failed, and is never followed by a recovery. An
issue carries the whole finding (observed vs threshold, the first
diagnostic step, the run and trace links), emails on open, is COMMENTED on
while the breach persists (one thread, not a daily email storm), and is
CLOSED with a comment when a run evaluates the check and it passes — the
recovery mail native notifications never send.

Identity is (check, case, detail), carried in an HTML comment marker in the
issue body rather than parsed out of the title: titles are edited by
people, markers are not. Recovery is scoped to what the run EVALUATED
(`evaluated`: case → checks): the daily run evaluates one case and must not
close the weekly cases' issues, and a fetch-only run must not close a
pipeline-tier issue. Silence about a check is not a recovery of it.

Transport is injectable so the tests drive a fake GitHub; the default uses
`requests` against api.github.com with the Actions `GITHUB_TOKEN`
(`permissions: issues: write` on the workflow).
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from evals.thresholds import Finding

LABEL = "canary"
LABEL_COLOR = "b60205"
LABEL_DESCRIPTION = "Opened and closed by the canary watcher"
MARKER_RE = re.compile(r"<!--\s*canary\s+check=(\S+)\s+case=(\S+)\s+detail=(\S*)\s*-->")

Transport = Callable[[str, str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]], Tuple[int, Any]]


class GitHubError(RuntimeError):
    pass


def marker(finding: Finding) -> str:
    return f"<!-- canary check={finding.check} case={finding.case_id} detail={finding.detail} -->"


def identity_of(issue: Mapping[str, Any]) -> Optional[Tuple[str, str, str]]:
    match = MARKER_RE.search(str(issue.get("body") or ""))
    return (match.group(1), match.group(2), match.group(3) or "") if match else None


def _stamp(now: Optional[datetime]) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")


def issue_body(finding: Finding, run_url: Optional[str], trace_url: Optional[str], now: Optional[datetime] = None) -> str:
    lines = [
        f"**{finding.title}**",
        "",
        f"- Observed: {finding.observed}",
        f"- Threshold: {finding.threshold}",
        f"- First step: {finding.first_step}",
        f"- Run: {run_url or 'n/a'}",
        f"- Trace: {trace_url or 'n/a'}",
        f"- Opened: {_stamp(now)}",
        "",
        "This issue is managed by the canary watcher: it is commented on while the breach "
        "persists and closed automatically when a run evaluates the check and it passes.",
        "",
        marker(finding),
    ]
    return "\n".join(lines)


def repeat_comment(finding: Finding, run_url: Optional[str], trace_url: Optional[str], now: Optional[datetime] = None) -> str:
    return (
        f"Still breached at {_stamp(now)}: observed {finding.observed} (threshold {finding.threshold}).\n"
        f"Run: {run_url or 'n/a'} · Trace: {trace_url or 'n/a'}"
    )


def recovery_comment(run_url: Optional[str], now: Optional[datetime] = None) -> str:
    return f"Recovered: the {_stamp(now)} run evaluated this check and it passed. Run: {run_url or 'n/a'}"


class GitHubIssues:
    """The handful of Issues API calls the watcher needs."""

    def __init__(self, token: str, repo: str, transport: Optional[Transport] = None, api_url: str = "https://api.github.com"):
        self.repo = repo
        self._call: Transport = transport or _requests_transport(token, api_url)

    def list_open(self, label: str = LABEL) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        for page in range(1, 6):  # 500 open canary issues would be its own alert
            status, data = self._call("GET", f"/repos/{self.repo}/issues", None, {
                "labels": label, "state": "open", "per_page": 100, "page": page,
            })
            if status >= 400:
                raise GitHubError(f"list issues: HTTP {status}")
            batch = [i for i in (data or []) if isinstance(i, dict) and not i.get("pull_request")]
            issues.extend(batch)
            if len(data or []) < 100:
                break
        return issues

    def create(self, title: str, body: str, labels: Sequence[str] = (LABEL,)) -> Dict[str, Any]:
        status, data = self._call("POST", f"/repos/{self.repo}/issues", {
            "title": title, "body": body, "labels": list(labels),
        }, None)
        if status >= 400:
            raise GitHubError(f"create issue: HTTP {status}")
        return data

    def comment(self, number: int, body: str) -> Dict[str, Any]:
        status, data = self._call("POST", f"/repos/{self.repo}/issues/{number}/comments", {"body": body}, None)
        if status >= 400:
            raise GitHubError(f"comment on #{number}: HTTP {status}")
        return data

    def close(self, number: int) -> Dict[str, Any]:
        status, data = self._call("PATCH", f"/repos/{self.repo}/issues/{number}", {
            "state": "closed", "state_reason": "completed",
        }, None)
        if status >= 400:
            raise GitHubError(f"close #{number}: HTTP {status}")
        return data

    def ensure_label(self, name: str = LABEL) -> bool:
        """True when the label had to be created."""
        status, _ = self._call("GET", f"/repos/{self.repo}/labels/{name}", None, None)
        if status == 200:
            return False
        if status != 404:
            raise GitHubError(f"read label {name}: HTTP {status}")
        status, _ = self._call("POST", f"/repos/{self.repo}/labels", {
            "name": name, "color": LABEL_COLOR, "description": LABEL_DESCRIPTION,
        }, None)
        if status >= 400:
            raise GitHubError(f"create label {name}: HTTP {status}")
        return True


def _requests_transport(token: str, api_url: str) -> Transport:
    import requests

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "carecompass-canary-watcher",
    })

    def call(method: str, path: str, json: Optional[Dict[str, Any]], params: Optional[Dict[str, Any]]) -> Tuple[int, Any]:
        response = session.request(method, f"{api_url}{path}", json=json, params=params, timeout=30)
        try:
            data = response.json() if response.content else None
        except ValueError:
            data = None
        return response.status_code, data

    return call


@dataclass
class SyncResult:
    opened: List[str] = field(default_factory=list)
    commented: List[str] = field(default_factory=list)
    closed: List[str] = field(default_factory=list)
    untouched: int = 0
    errors: List[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"issues: opened {len(self.opened)}, commented {len(self.commented)}, "
            f"closed {len(self.closed)}, left open (not evaluated this run) {self.untouched}"
            + (f", errors {len(self.errors)}" if self.errors else "")
        )


def sync_issues(
    client: GitHubIssues,
    findings: Sequence[Finding],
    evaluated: Mapping[str, Iterable[str]],
    run_url: Optional[str] = None,
    trace_urls: Optional[Mapping[str, Optional[str]]] = None,
    label: str = LABEL,
    now: Optional[datetime] = None,
) -> SyncResult:
    """Open / comment / close so the open issues mirror the current breaches.

    `evaluated` maps case_id → the checks this run evaluated for it; an open
    issue outside that map is left alone. A failed API call on one issue is
    recorded and the rest proceed — but a failed LISTING aborts, because
    creating without knowing what is open would duplicate every issue.
    """
    result = SyncResult()
    trace_urls = trace_urls or {}
    try:
        client.ensure_label(label)
    except GitHubError as exc:
        result.errors.append(str(exc))
    open_issues = client.list_open(label)
    by_identity: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for issue in open_issues:
        key = identity_of(issue)
        if key and key not in by_identity:
            by_identity[key] = issue
    current = set()
    for finding in findings:
        key = finding.identity
        if key in current:
            continue  # two findings with one identity (should not happen) → one issue
        current.add(key)
        trace = trace_urls.get(finding.case_id)
        try:
            if key in by_identity:
                client.comment(int(by_identity[key]["number"]), repeat_comment(finding, run_url, trace, now))
                result.commented.append(finding.title)
            else:
                client.create(finding.title, issue_body(finding, run_url, trace, now), [label])
                result.opened.append(finding.title)
        except GitHubError as exc:
            result.errors.append(f"{finding.title}: {exc}")
    for key, issue in by_identity.items():
        if key in current:
            continue
        check, case_id, _ = key
        if check in set(evaluated.get(case_id, ())):
            try:
                number = int(issue["number"])
                client.comment(number, recovery_comment(run_url, now))
                client.close(number)
                result.closed.append(str(issue.get("title") or number))
            except GitHubError as exc:
                result.errors.append(f"close {issue.get('title')}: {exc}")
        else:
            result.untouched += 1
    return result
