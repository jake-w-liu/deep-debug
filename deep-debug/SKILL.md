---
name: deep-debug
description: Iterative deep debugging workflow for finding, verifying, fixing, and rechecking real software bugs. Use when the user asks for deep-debug, deep debugging, a fresh bug audit, a bug hunt, an audit, verify, fix, and reverify loop, or wants Codex to keep investigating until no confirmed bugs remain while avoiding false positives.
---

# Deep Debug

## Workflow

Run a fresh iterative bug hunt:

1. Rebuild context from the current repo, user request, failing behavior, logs, tests, and recent diffs. Do not rely on prior assumptions when starting a new pass.
2. Audit for plausible bugs from several angles: user workflows, data boundaries, error paths, concurrency, configuration, persistence, integrations, and test coverage gaps.
3. Turn each candidate into a falsifiable claim with expected behavior, actual behavior, affected code path, and evidence.
4. Verify before fixing. Prefer a failing test, reproduction command, runtime trace, typecheck/build failure, or direct control-flow/data-flow proof. Discard candidates that cannot survive verification.
5. Fix only confirmed bugs. Keep changes local, follow existing patterns, preserve unrelated user changes, and add or update regression tests when feasible.
6. Reverify the original failure, the new/updated tests, and nearby or broader checks proportional to risk.
7. Repeat from a fresh audit pass until no new verified bugs are found.

## Goal Handling

When goal tools are available, create a goal before starting only if the user explicitly asks to set the deep-debug run as a goal. Treat prompts like `Use $deep-debug and set this as a goal...` or `Use deep-debug as a goal...` as explicit goal requests.

Do not create a goal from `$deep-debug` alone. Goal creation requires an explicit user request.

If a goal is created, use an objective that names the target and includes the full loop: audit, verify, fix, and reverify until no confirmed bugs remain. Mark the goal complete only after the stop condition is satisfied.

## False-Positive Guardrails

- Do not present style issues, speculative improvements, missing features, or harmless edge cases as bugs unless they violate an explicit requirement or observable behavior.
- Separate confirmed bugs from unverified risks and test gaps. Fix confirmed bugs; report unverified risks only when they matter and cannot be checked within the available constraints.
- Check whether suspicious code is intentional compatibility behavior, generated output, test-only scaffolding, feature-flagged behavior, platform-specific behavior, or covered by callers before calling it a bug.
- If verification needs current external facts, installed package behavior, or service documentation, look it up or reproduce locally instead of guessing.

## Fresh-Pass Tactics

- On each iteration, change the audit angle instead of only re-reading the previous finding: follow a different entry point, inspect neighboring modules, run a different test scope, or trace a different data shape.
- When multi-agent tools are available, current instructions permit delegation, and the task is broad enough, launch an independent fresh audit with only the task-relevant context and compare concrete evidence, not conclusions.
- Keep a compact ledger of candidates: `candidate`, `verification`, `status`, `fix`, and `reverify`.

## Stop Condition

Stop only after:

- all confirmed bugs found so far are fixed or explicitly blocked,
- each fix has been reverified,
- one final fresh audit pass finds no new verified bugs, and
- the final response states what was fixed, what was run, and any remaining unverified risks or test gaps.
