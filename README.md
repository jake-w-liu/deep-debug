# deep-debug

Codex skill for iterative deep debugging across the full applicable problem surface:

1. run a fresh from-scratch audit,
2. verify suspected bugs before treating them as real,
3. cover domain correctness, implementation correctness, memory/resource behavior, and performance where relevant,
4. fix confirmed issues,
5. CRC-reverify every fix,
6. repeat until a final fresh pass finds no confirmed bugs.

The updated skill is designed for users who want stronger assurance on:

- math and physics correctness when the target contains domain logic,
- code implementation correctness,
- memory allocation and resource efficiency,
- performance and optimization claims backed by measurement rather than guesswork.

It does not permit unverified claims such as "fully optimized" or "physically correct". Those claims now require evidence, or the remaining gap must be stated explicitly.

## Install

Recommended: install with Codex's skill installer:

```text
$skill-installer https://github.com/jake-w-liu/deep-debug/tree/main/deep-debug
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
Use $deep-debug and set this as a goal to audit this code path from scratch for real bugs across domain correctness, implementation correctness, memory/resource behavior, and performance where applicable; verify each finding, fix confirmed issues, and CRC-reverify until no confirmed bugs remain.
```

It is useful for requests like:

- `Use $deep-debug and set this as a goal on this failing workflow.`
- `Deep-debug this PR and fix real bugs only.`
- `Run a fresh audit, verify findings, fix, and reverify.`
- `Use $deep-debug for a from-scratch audit of math correctness, implementation correctness, memory behavior, and performance, then fix and CRC-reverify everything you can prove.`

Goal creation still needs to be explicit. `$deep-debug` alone should run the workflow, but a prompt that says `set this as a goal` should create and complete a goal around the full audit, fix, and reverify loop.

## Files

- `deep-debug/SKILL.md`: installable skill instructions.
- `deep-debug/agents/openai.yaml`: installable skill UI metadata.
- Root `SKILL.md` and `agents/openai.yaml`: keep direct-clone installs valid.
