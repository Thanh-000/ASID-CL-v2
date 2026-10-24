from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "main.tex"
BIB = ROOT / "references.bib"


def strip_comments(text: str) -> str:
    cleaned: list[str] = []
    for line in text.splitlines():
        cut = len(line)
        for index, char in enumerate(line):
            if char != "%":
                continue
            slash_count = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                slash_count += 1
                cursor -= 1
            if slash_count % 2 == 0:
                cut = index
                break
        cleaned.append(line[:cut])
    return "\n".join(cleaned)


def included_tex_files() -> tuple[list[Path], list[str]]:
    pending = [MAIN]
    found: list[Path] = []
    errors: list[str] = []
    seen: set[Path] = set()
    while pending:
        path = pending.pop(0).resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            errors.append(f"Missing TeX input: {path.relative_to(ROOT)}")
            continue
        found.append(path)
        text = strip_comments(path.read_text(encoding="utf-8"))
        for input_name in re.findall(r"\\(?:input|include)\{([^}]+)\}", text):
            candidate = ROOT / input_name
            if candidate.suffix == "":
                candidate = candidate.with_suffix(".tex")
            pending.append(candidate)
    return found, errors


def brace_errors(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    depth = 0
    for index, char in enumerate(text):
        if char not in "{}":
            continue
        slash_count = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            slash_count += 1
            cursor -= 1
        if slash_count % 2 == 1:
            continue
        depth += 1 if char == "{" else -1
        if depth < 0:
            errors.append(f"{path.name}: closing brace before opening brace")
            return errors
    if depth:
        errors.append(f"{path.name}: unbalanced braces (net depth {depth})")
    return errors


def environment_errors(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    tokens = re.finditer(r"\\(begin|end)\{([^}]+)\}", text)
    stack: list[str] = []
    for token in tokens:
        action, environment = token.groups()
        if action == "begin":
            stack.append(environment)
            continue
        if not stack:
            errors.append(f"{path.name}: orphan end{{{environment}}}")
            continue
        expected = stack.pop()
        if expected != environment:
            errors.append(
                f"{path.name}: end{{{environment}}} closes begin{{{expected}}}"
            )
    if stack:
        errors.append(f"{path.name}: unclosed environments: {', '.join(stack)}")
    return errors


def main() -> int:
    tex_files, errors = included_tex_files()
    if not BIB.is_file():
        errors.append("Missing references.bib")
        bib_text = ""
    else:
        bib_text = BIB.read_text(encoding="utf-8")

    clean_texts: dict[Path, str] = {}
    for path in tex_files:
        text = path.read_text(encoding="utf-8")
        clean = strip_comments(text)
        clean_texts[path] = clean
        errors.extend(brace_errors(path, clean))
        errors.extend(environment_errors(path, clean))
        for marker in ("Ã", "Â", "Ä\u0091", "áº", "á»"):
            if marker in text:
                errors.append(f"{path.name}: possible mojibake marker {marker!r}")

    joined = "\n".join(clean_texts.values())
    label_counts = Counter(re.findall(r"\\label\{([^}]+)\}", joined))
    duplicate_labels = sorted(label for label, count in label_counts.items() if count > 1)
    if duplicate_labels:
        errors.append(f"Duplicate labels: {', '.join(duplicate_labels)}")

    references = set(
        re.findall(r"\\(?:ref|eqref|autoref|cref|Cref)\{([^}]+)\}", joined)
    )
    missing_labels = sorted(references - set(label_counts))
    if missing_labels:
        errors.append(f"Undefined labels: {', '.join(missing_labels)}")

    citation_groups = re.findall(
        r"\\cite(?:p|t|author|year|yearpar|alp|alt)?\*?\{([^}]+)\}", joined
    )
    citations = {
        key.strip()
        for group in citation_groups
        for key in group.split(",")
        if key.strip()
    }
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_text))
    missing_citations = sorted(citations - bib_keys)
    if missing_citations:
        errors.append(f"Missing bibliography keys: {', '.join(missing_citations)}")

    forbidden = re.findall(r"(?i)phase[\s-]*4", joined)
    if forbidden:
        errors.append("The report still contains a removed phase reference")

    if re.search(r"\\mf_", joined):
        errors.append(r"Unsafe subscript on \\mf macro; write F_{1,x}^{macro} explicitly")

    print(f"TeX files checked: {len(tex_files)}")
    print(f"Labels: {len(label_counts)}; references: {len(references)}")
    print(f"Citations used: {len(citations)}; bibliography entries: {len(bib_keys)}")
    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Static Overleaf validation: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
