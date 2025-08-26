import re
from typing import List, Tuple


CITATION_RE = re.compile(r"\[(\d+)\]")


def validate_and_fix_citations(answer: str, passages: List) -> Tuple[str, bool, List[int]]:
    max_id = len(passages)
    phantom: List[int] = []
    mapping = {}
    next_id = 1

    def repl(match: re.Match) -> str:
        nonlocal next_id
        num = int(match.group(1))
        if 1 <= num <= max_id:
            if num not in mapping:
                mapping[num] = next_id
                next_id += 1
            return f"[{mapping[num]}]"
        phantom.append(num)
        return ""

    fixed = CITATION_RE.sub(repl, answer)
    return fixed, bool(phantom), phantom
