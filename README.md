# agentpair

Does a second model reviewing an agent's code make the code better, and is it worth what it
costs? This is a harness that measures it, and the numbers it produced over 20 tasks.

## The two arms

**pair**: an implementer writes a module and its tests from a spec. A reviewer, running as a
separate session, raises findings with evidence. The implementer applies what it accepts and
rejects the rest with a reason. Every rejection goes to an orchestrator, which decides who is
right, running code first when running something would settle it. Where it rules for the
reviewer, the implementer is sent back once to carry out what it lost. One pass, no loop.

**baseline**: one agent writes the code and its tests and declares itself done.

Every role in a run uses the same model unless told otherwise, so the only difference between
the arms is the review mechanism rather than model capability. Both arms are then scored
against held-out tests the agents never see.

## Results

20 BigCodeBench tasks, 102 held-out tests. Every task's reference solution passes all of its
own tests, so no failure below is the benchmark's fault.

| implementer | reviewer | pair | baseline | gap | pair cost | baseline cost | cost ratio | pair time | baseline time |
|---|---|---|---|---|---|---|---|---|---|
| sonnet | opus | 78.4% | 70.6% | **+7.8** | $10.38 | $2.39 | 4.34x | 44.8 min | 10.7 min |
| opus | opus | 91.2% | 79.4% | **+11.8** | $15.15 | $4.62 | 3.28x | 51.4 min | 16.3 min |

Models: opus is Claude Opus 5, sonnet is Claude Sonnet 5.

Times are wall clock for all 20 tasks run one after another, so they are what the sweep takes
end to end rather than per task.

An opus reviewer is worth what it costs. Reviewing opus's own code it adds nearly 12 points,
and reviewing sonnet's it adds nearly 8, for roughly three to four times the price of the
single-agent run.

The finding counts track the gap: 42 findings across 20 tasks with opus reviewing opus, and 62
with opus reviewing sonnet.

Worth stating against our own result: sonnet+opus reaches 78.4% for $10.38, while opus writing
alone reaches 79.4% for $4.62. On this benchmark, buying an expensive reviewer for cheap code
costs more than simply using the expensive model to write it. The configuration that wins is
opus reviewing opus.

The cost and time ratios are also less lopsided than they look, because they compare one
baseline attempt against one reviewed attempt. Nobody ships the first draft of a failing task.
To reach the pair arm's score a single agent would be run again on what it got wrong, and
every one of those attempts costs roughly what the first did, so a baseline that is retried
until it matches converges on the pair arm's cost and wall clock. We have not measured how
many attempts that takes, so the numbers above are stated as they were run: one attempt each.
Read the ratios as an upper bound on the premium, not as the premium.

Disputes are rare. Across these 40 runs the implementer rejected a finding 5 times, and the
orchestrator ruled against it once, which is the only time the remediation turn has fired.

## Install

```bash
uv sync
```

## Run one task

```bash
uv run agentpair --task path/to/spec.md --module solution.py --arm both --judge reviewer
```

`--judge` is required for the pair arm and has no default, because the orchestrator decides
who is right and the command should not pick that on your behalf. Pass `--judge reviewer` to
reuse the reviewer's model, which adds no third model to the run, or `--judge <model>` to have
a third party rule. The model that ruled is always recorded. When every role shares one model,
the reviewer is necessarily the judge, and that is a property of the design rather than an
oversight.

Each run is recorded and workspaced under a task id, which is the task file's stem by default.
A benchmark keeps every task's spec in a `spec.md` inside a directory named for the task, so a
generic stem takes the directory's name instead, and `--task-id` overrides both.

## Run the benchmark

```bash
uv run python -u scripts/run_benchmark.py \
  --tasks-dir benchmark/bigcodebench \
  --results-dir /path/outside/this/repo \
  --model opus
```

Sequential on purpose: wall clock is one of the measured quantities, and concurrent runs
competing for CPU and network inflate every duration. Resumable, since a `(task, arm)` pair
already in the results file is skipped. `--implementer-model` and `--reviewer-model` override
the single model for a mixed round.

Point `--results-dir` somewhere outside this repository. The agents work in that directory with
ordinary filesystem access, so the further the answers are from where they are working, the
less the results depend on trusting them.

## Score it

```bash
uv run python -u scripts/score_benchmark.py \
  --results-dir /path/outside/this/repo \
  --answers-dir /path/to/answers
```

Each task's reference solution is scored first. A held-out test its own reference cannot pass
is broken, and every agent measured against it would be marked wrong for the benchmark's
mistake, so such a task is reported and excluded.

## The held-out tests

Each task directory carries three files:

- `spec.md`: the task description. This is the only one the agents are given.
- `hidden_test.py`: the benchmark's own tests, used to score a solution after the run.
- `reference.py`: BigCodeBench's canonical solution, used to prove the held-out test is
  sound before any agent is measured against it.

The agents are never shown the last two. The harness passes only the text of `spec.md` as the
prompt, and scoring copies the solution and the held-out test into a throwaway directory, so
neither file is ever written into a run workspace.

They are in the repository because a benchmark whose tests you cannot read is a benchmark you
cannot check. The cost is that keeping runs honest now depends on `--results-dir` pointing
outside this repository: the agents work in that directory with ordinary filesystem access,
and nothing but distance stops one from reading the answers. Every number in this README was
produced with the run directories on a separate path and the held-out tests not present in the
tree at the time.

Regenerate them from source if you would rather not trust the copies here:

```bash
uv run python scripts/fetch_bigcodebench.py --out-dir benchmark/bigcodebench
```

It downloads BigCodeBench, rewrites each test to import `solution`, and checks every task by
running its held-out tests against its own reference before keeping it.

## What the record says

Every finished run appends one JSON line to the results file, carrying:

- when it started, and the model used for each role
- every finding the reviewer raised, with its evidence
- every finding the implementer accepted, with the reason it gave
- every finding it rejected, how the orchestrator settled it, and whether a probe was run
- which findings were sent back to be applied after a ruling
- what the agent's own suite did
- tokens, split into cached and uncached, with cost and wall clock

A degraded run never reads like a clean one, because each of these is recorded with a reason
rather than silently absorbed: findings the implementer never answered, disputes the
orchestrator could not settle, decisions naming a finding that does not exist, disputes ruled
on after a probe could not be produced, and turns cut short by a budget or turn cap.

Two things the record deliberately does not claim. `sent_back_after_ruling` says the
implementer was asked to apply a ruling, not that the files changed, because nothing here
checks the edit. And the agent's own suite result measures whether the agent agreed with
itself, which is why the score comes from the held-out tests instead.

All four model turns answer in a `json` fence: the reviewer's findings, the implementer's
decisions, and the orchestrator's probe and ruling. Nothing in the harness reads a decision
out of model prose. For the same reason the agent's pass and fail counts are read from
pytest's JUnit XML report rather than its terminal summary, which is prose a test can imitate
by printing it.

## Tests

```bash
uv run pytest -q
uv run ruff check src tests scripts
```
