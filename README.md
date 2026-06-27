# deep-debug

Codex skill for iterative deep debugging:

1. run a fresh bug audit,
2. verify suspected bugs before treating them as real,
3. fix confirmed issues,
4. reverify,
5. repeat until a final fresh pass finds no confirmed bugs.

## Install

Recommended: ask Codex to install the skill with `$skill-installer`:

```text
Use $skill-installer to install https://github.com/jake-w-liu/deep-debug/tree/main/deep-debug
```

Restart Codex after installing so the new skill is picked up.

If you already have a local `deep-debug` skill folder, move it aside before installing. The installer aborts when the destination already exists.

## Manual install

You can also clone this repository directly into your Codex skills directory:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
git clone https://github.com/jake-w-liu/deep-debug.git "${CODEX_HOME:-$HOME/.codex}/skills/deep-debug"
```

## Update

For `$skill-installer` installs, reinstall from the URL above after moving aside the old local copy.

For manual clone installs:

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

- `deep-debug/SKILL.md`: installable skill instructions.
- `deep-debug/agents/openai.yaml`: installable skill UI metadata.
- Root `SKILL.md` and `agents/openai.yaml`: keep direct-clone installs valid.
