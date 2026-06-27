# deep-debug

Codex skill for iterative deep debugging:

1. run a fresh bug audit,
2. verify suspected bugs before treating them as real,
3. fix confirmed issues,
4. reverify,
5. repeat until a final fresh pass finds no confirmed bugs.

## Install

Clone this repository into your Codex skills directory:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
git clone https://github.com/jake-w-liu/deep-debug.git "${CODEX_HOME:-$HOME/.codex}/skills/deep-debug"
```

If you already have a local `deep-debug` skill folder, move it aside or update it manually before cloning.

## Update

```bash
cd "${CODEX_HOME:-$HOME/.codex}/skills/deep-debug"
git pull
```

## Use

Invoke the skill explicitly:

```text
Use $deep-debug to audit this code path for real bugs, verify each finding, fix confirmed issues, and reverify until no bugs remain.
```

It is useful for requests like:

- `Use $deep-debug on this failing workflow.`
- `Deep-debug this PR and fix real bugs only.`
- `Run a fresh audit, verify findings, fix, and reverify.`

## Files

- `SKILL.md`: skill instructions loaded by Codex.
- `agents/openai.yaml`: UI metadata for the skill.
