import re
import io
import tokenize
import sys
from pathlib import Path

NB_PATH = Path(sys.argv[1])
BACKUP_PATH = NB_PATH.with_suffix(NB_PATH.suffix + '.bak')

print(f"Notebook: {NB_PATH}")
print(f"Backup: {BACKUP_PATH}")

text = NB_PATH.read_text(encoding='utf-8')

pattern = re.compile(r"(<VSCode.Cell[^>]*language=\"python\"[^>]*>)(.*?)(</VSCode.Cell>)", re.S)

def remove_comments_from_code(code: str) -> str:
    # Use tokenize to remove comments while preserving code and string literals
    try:
        buf = io.StringIO(code)
        out_tokens = []
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        tokgen = tokenize.generate_tokens(buf.readline)
        for tok_type, tok_string, (srow, scol), (erow, ecol), line in tokgen:
            if tok_type == tokenize.COMMENT:
                continue
            out_tokens.append((tok_type, tok_string))
        # Untokenize
        return tokenize.untokenize(out_tokens)
    except Exception as e:
        print(f"Tokenize error: {e}")
        # Fallback: remove lines that start with # and inline comments naively
        new_lines = []
        for ln in code.splitlines():
            s = ln
            stripped = s.lstrip()
            if stripped.startswith('#'):
                continue
            if '#' in s:
                # naive remove inline comment
                parts = s.split('#')
                # keep part before '#'
                s = parts[0].rstrip()
            new_lines.append(s)
        return '\n'.join(new_lines)

new_text = text
changes = 0
for m in pattern.finditer(text):
    whole = m.group(0)
    open_tag = m.group(1)
    code = m.group(2)
    close_tag = m.group(3)

    # remove leading/trailing newlines
    code_stripped = code
    cleaned = remove_comments_from_code(code_stripped)

    if cleaned != code_stripped:
        changes += 1
        replacement = open_tag + cleaned + close_tag
        new_text = new_text.replace(whole, replacement)

if changes == 0:
    print("No changes needed (no python comments found or already cleaned).")
else:
    print(f"Modified {changes} python cell(s). Writing backup and updating notebook...")
    NB_PATH.rename(BACKUP_PATH)
    NB_PATH.write_text(new_text, encoding='utf-8')
    print("Done.")
