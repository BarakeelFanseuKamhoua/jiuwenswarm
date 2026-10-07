# Designer P0 fix — M15 (shot delete / chat edit)

**Branch:** `0.2.8.beta1-A-P0-M15Fixes-1.0`  
**Base:** `0.2.8.beta1`  
**Scope:** M15 only — duration false-positive + missing-shot scrub after canvas delete.  
**Out of scope:** M08 toolbar / `prompt_origin=user` WAN authority (separate branch later).

## Problem

After canvas-deleting shot 3, chat-editing shot 2 fails with:

`Document n_brief references missing shots: [3, 5]`

Root causes:

1. Canvas delete leaves stale brief / `镜头 3` prose.
2. `_SHOT_REFERENCE` false-positive: `每个镜头 5 秒` → shot `5`.

## Fix

1. **Regex filter** — duration / distributive idioms are not shot indices.
2. **Tombstone + deterministic scrub** in `prepare_document_update` before fail-closed validate; prior-canvas missing indices via `missing_shot_tombstones` in document edit context.

## Files

- `chat_shot_references.py` — grammar + `scrub_missing_shot_references`
- `chat_document_sync.py` — scrub before validate
- `chat_document_plan.py` — tombstone missing indices against live candidate set

## Tests

- `test_shot_reference_grammar.py`
- `test_missing_shot_scrub.py`
- `test_m15_pipeline_scenarios.py`
