"""Check this source-only snapshot without importing experiment dependencies."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '__pycache__', '.venv', 'venv'}
PATTERNS = {
    'private key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'GitHub token': r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b',
    'Hugging Face token': r'\bhf_[A-Za-z0-9]{20,}\b',
    'AWS access key': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'assigned credential': r'''(?i)(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)["']?\s*[:=]\s*["'][A-Za-z0-9_+/=-]{16,}["']''',
    'credential in URL': r'https?://[^\s/:]+:[^\s/@]+@',
    'token in URL': r'(?i)[?&](?:access_token|accessToken|api_key|token)=[A-Za-z0-9_-]{16,}',
}


def main():
    errors = []
    counts = {'files': 0, 'python': 0, 'json': 0}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or any(p in EXCLUDED for p in path.relative_to(ROOT).parts):
            continue
        rel = path.relative_to(ROOT).as_posix()
        counts['files'] += 1
        try:
            text = path.read_text(encoding='utf-8-sig')
            if path.suffix == '.py':
                ast.parse(text, filename=rel)
                counts['python'] += 1
            elif path.suffix == '.json':
                json.loads(text)
                counts['json'] += 1
        except (UnicodeError, SyntaxError, ValueError) as exc:
            errors.append(f'{rel}: {exc}')
            continue
        for name, pattern in PATTERNS.items():
            if re.search(pattern, text):
                errors.append(f'{rel}: possible {name}; inspect before publishing')
        if path.name == 'README.md' or path.parent.name == 'docs' and path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if '://' in target or target.startswith('#'):
                    continue
                if not (path.parent / target.split('#')[0]).exists():
                    errors.append(f'{rel}: missing linked file {target}')
    manifest = json.loads((ROOT / 'docs/SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = ROOT / item['path']
        if not path.is_file():
            errors.append(f'missing source: {item["path"]}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item['shared_sha256']:
            errors.append(f'source hash differs: {item["path"]}')
    print(json.dumps({'status': 'FAIL' if errors else 'PASS', **counts,
                      'source_files': len(manifest['files']), 'errors': errors}, indent=2))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
