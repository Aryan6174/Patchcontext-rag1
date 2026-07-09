from typing import List, Dict
from dataclasses import dataclass
from langchain_core.documents import Document


@dataclass
class PatchContextResponse:
    answer: str
    citations: Dict
    is_verified: bool
    verification_score: float
    warning: str = ""

    def format_response(self):
        output = f"**Answer:**\n\n{self.answer}\n\n"

        if self.warning:
            output += f"⚠️ *{self.warning}*\n\n"

        output += "---\n\n**Supporting Evidence:**\n\n"

        if self.citations["commits"]:
            output += "**Commits:**\n"
            for c in self.citations["commits"]:
                output += f"- [{c['sha']}]({c['url']})\n"
            output += "\n"

        if self.citations["pull_requests"]:
            output += "**Pull Requests:**\n"
            for pr in self.citations["pull_requests"]:
                output += f"- [PR #{pr['number']}: {pr['title']}]({pr['url']})\n"
            output += "\n"

        if self.citations["issues"]:
            output += "**Issues:**\n"
            for issue in self.citations["issues"]:
                output += f"- [Issue #{issue['number']}: {issue['title']}]({issue['url']})\n"
            output += "\n"

        output += f"\n*Verification Score: {self.verification_score:.2f}*"
        return output


def format_context(documents):
    parts = []
    for idx, doc in enumerate(documents, 1):
        meta = doc.metadata
        section = f"--- Source {idx} ---\nType: {meta.get('type', 'unknown')}\n"

        if meta.get("type") == "commit":
            section += f"Commit SHA: {meta.get('short_sha', 'N/A')}\nURL: {meta.get('url', 'N/A')}\n"
        elif meta.get("type") == "pull_request":
            section += f"PR Number: #{meta.get('pr', 'N/A')}\nURL: {meta.get('url', 'N/A')}\n"
        elif meta.get("type") == "issue":
            section += f"Issue Number: #{meta.get('issue', 'N/A')}\nURL: {meta.get('url', 'N/A')}\n"

        section += f"\nContent:\n{doc.page_content}\n"
        parts.append(section)

    return "\n".join(parts)


def extract_citations(documents):
    citations = {"commits": [], "pull_requests": [], "issues": []}
    seen_commits, seen_prs, seen_issues = set(), set(), set()

    for doc in documents:
        meta = doc.metadata

        if meta.get("type") == "commit":
            sha = meta.get("short_sha")
            if sha and sha not in seen_commits:
                citations["commits"].append({
                    "sha": sha,
                    "url": meta.get("url", "")
                })
                seen_commits.add(sha)

        elif meta.get("type") == "pull_request":
            pr_num = meta.get("pr")
            if pr_num and pr_num not in seen_prs:
                citations["pull_requests"].append({
                    "number": pr_num,
                    "title": meta.get("title", ""),
                    "url": meta.get("url", "")
                })
                seen_prs.add(pr_num)

        elif meta.get("type") == "issue":
            issue_num = meta.get("issue")
            if issue_num and issue_num not in seen_issues:
                citations["issues"].append({
                    "number": issue_num,
                    "title": meta.get("title", ""),
                    "url": meta.get("url", "")
                })
                seen_issues.add(issue_num)

    return citations


def clean_answer(text):
    text = text.replace("Pydantric", "Pydantic")
    text = text.replace("opeanapi", "OpenAPI")

    sentences = text.split('. ')
    last = sentences[-1].strip()

    if len(sentences) > 1 and (len(last) < 15 or last[-1] not in '.!?'):
        sentences = sentences[:-1]

    cleaned = '. '.join(sentences)

    if cleaned and cleaned[-1] not in '.!?':
        cleaned += '.'

    return cleaned


def compute_verification_score(answer, context):
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    if not answer_words:
        return 0.0

    overlap = answer_words & context_words
    score = len(overlap) / len(answer_words)

    return round(min(score, 1.0), 2)