# /npu-discuss-op

Resolve ambiguity in an operator specification before planning begins.

## Usage

```
/npu-discuss-op <op_name>
```

## Arguments

- `op_name` — The operator name as registered in `OPERATOR_REGISTRY.yaml` and `.npu-harness/operators/<op_name>/`.

## Preconditions

- `.npu-harness/operators/<op_name>/SPEC.yaml` exists (created by `/npu-new-op`)
- `.npu-harness/operators/<op_name>/DECISIONS.md` exists

## What to Read Before Discussing

1. `.npu-harness/operators/<op_name>/SPEC.yaml` — The current spec
2. `.npu-harness/operators/<op_name>/DECISIONS.md` — Existing decisions and assumptions
3. `OPERATOR_REGISTRY.yaml` — Context on where this operator fits

## Workflow

### Step 1: Audit the Spec for Ambiguity

Review every section of SPEC.yaml and identify any field that is:

- Missing entirely
- Present but empty (e.g., `shape_symbols: {}`)
- Marked with a placeholder like `TODO`, `TBD`, or `?`
- Contradictory with another field
- Vague enough to cause implementation ambiguity

Check specifically:

| Section | Ambiguity Check |
|---------|----------------|
| name | Is it unique? Does it match the directory name? |
| category | Is the category correct for the operation? |
| math.definition | Can the math be implemented from this description alone? |
| math.equations | Are equations precise? Is notation consistent? |
| inputs | Are all dtypes listed? Are shapes fully specified? |
| outputs | Are output shapes derivable from input shapes? |
| parameters | Are defaults sensible? Are types correct? |
| shape_symbols | Are symbol names consistent across inputs/outputs? |
| shape_rules | Are constraints complete? Are constraints satisfiable? |
| dtype_rules | Is accumulation dtype specified? Is casting behavior clear? |
| broadcasting | If supported, are rules precise? |
| masking | If supported, are types and alignment rules defined? |
| accuracy | Are tolerances realistic for the operation and dtype? |
| performance | If required, are targets measurable? |
| tests | Are shapes sufficient? Are edge cases identified? |
| implementation | Is the plan clear about what is expected? |

### Step 2: Categorize Each Finding

For each ambiguity found, classify it as:

**INFER** — The detail can be safely inferred from context, conventions, or defaults. Examples:
- Missing test shapes: derive from shape_symbols
- Missing default tolerance: use Harness defaults from config
- Missing target arch: default to dav-2201
- Missing warmup/iterations: use benchmark defaults

**ASK** — The detail materially affects correctness, ABI, shape rules, dtype behavior, memory format, or performance target and cannot be inferred. Examples:
- Unclear whether output includes an optional bias term
- Conflicting dtype requirements between inputs
- The user mentioned two different normalization formulas
- Unclear whether broadcasting is row-wise or element-wise
- Masking alignment rule is ambiguous (different conventions give different results)
- Performance target that seems physically impossible for the hardware

### Step 3: Resolve INFER Items

For each INFER finding:

1. Infer the most reasonable value based on Harness conventions and the operator context.
2. Record the inference in `.npu-harness/operators/<op_name>/DECISIONS.md`:

```markdown
## Inferred Assumptions

### Assumption: <short title>
- **Field:** <SPEC.yaml path, e.g., `inputs[0].dtype`>
- **Inferred value:** <the value chosen>
- **Rationale:** <why this is the most reasonable choice>
- **Risk:** <low|medium|high>
- **Reversible:** <yes|no — can the user override later?>
```

3. Update SPEC.yaml with the inferred value.

### Step 4: Present ASK Items to User

For each ASK finding, present the user with:

- The SPEC.yaml field that is ambiguous
- The specific question that needs answering
- The candidate answers with tradeoffs (if applicable)
- A recommended default with rationale

Format each question clearly:

```
## Question <N>: <topic>

**Field:** <SPEC.yaml path>
**Issue:** <description of the ambiguity>
**Impact:** <what implementation decisions depend on this answer>

Options:
1. <option A> — <implications>
2. <option B> — <implications>

**Recommendation:** <option> — <rationale>
```

Ask all questions together rather than one at a time. Let the user answer in their own order.

### Step 5: Incorporate User Answers

After the user responds:

1. Update SPEC.yaml with the user's choices.
2. Record each decision in DECISIONS.md:

```markdown
## User Decisions

### Decision: <short title>
- **Date:** <today>
- **Question:** <what was asked>
- **Choice:** <what the user chose>
- **Rationale:** <user's reasoning, if provided>
```

3. Re-validate that the spec is now internally consistent.

### Step 6: Final Validation

After all ambiguities are resolved, verify:

- No remaining TODO/TBD/empty fields in critical sections
- No contradictory shape rules
- Performance targets are coherent (target is plausible for the hardware)
- Accuracy tolerances are appropriate for the dtype and operation
- All inputs and outputs have complete dtype and shape specifications

### Step 7: Report

Print a summary:

```
Discussion complete for <op_name>.

INFER: <N> assumptions inferred and recorded
ASK:   <M> questions resolved with user input

Updated files:
  .npu-harness/operators/<op_name>/SPEC.yaml
  .npu-harness/operators/<op_name>/DECISIONS.md

Next command: /npu-plan-op <op_name>
```

If there are still unresolved ASK items because the user deferred them, list them explicitly and note that they must be resolved before `/npu-execute-op`.

## Rules

1. Do NOT ask the user about things that can be inferred from Harness defaults or conventions.
2. Do NOT ask the user about things that have no implementation impact.
3. Do NOT ask yes/no questions that you could answer yourself by reading the spec more carefully.
4. DO ask when two reasonable engineers could make different choices and both would produce correct-but-different results.
5. DO ask when the answer determines whether a performance target is realistic.
6. DO ask when the answer changes the ABI (kernel parameter list, memory layout).
7. Record ALL inferences, not just the ones that might be wrong. Transparency matters.
8. Every inference must have a rationale. No guessing without explanation.

## Output

The final output is:
- Updated `SPEC.yaml` with all resolvable fields filled
- Updated `DECISIONS.md` with all inferences and user decisions recorded
- A clear signal whether planning can begin or whether unresolved questions remain
