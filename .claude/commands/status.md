---
description: Print the current project's status, budget, and progress.
allowed-tools:
  - Bash(python:*)
---

# /status

Print compact project state.

## Procedure

```python
from pathlib import Path
import json
from lib.project import project_status
from lib.state import load_run_state
from lib.workspace import resolve_workspace

proj = Path('.').resolve()
ws = resolve_workspace(None, start=proj.parent.parent)
info = project_status(ws, proj.name)
rs = load_run_state(proj)
print(json.dumps({"project": info, "run_state": rs.to_dict()}, indent=2, default=str))
```

Render to the user as a small table.
