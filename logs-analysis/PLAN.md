# PLAN.md — LLM Behavioral Metrics for Interactive Fiction Logs

## Overview

This project analyzes the behavior of a Large Language Model (LLM) interacting with a constrained system (Infocom-style parser).  
The goal is to **quantify limitations, repetitions, and cognitive patterns** in the model’s action proposals.

We treat logs as empirical traces of LLM behavior and extract metrics that reveal:

- lack of state tracking
- repetitive loops
- mismatch between reasoning and action
- exploration vs stagnation

## Input Data

### Log format

Logs contain sequences of:

- game outputs
- LLM thoughts (`AI thinks : ...`)
- LLM suggested commands (`AI suggests: 'COMMAND'`)

Example:

AI thinks : 'The ladder is likely the escape route'
AI suggests: 'CLIMB LADDER'

## Goals

1. Parse logs into structured data
2. Extract action sequences
3. Compute behavioral metrics
4. Output quantitative summaries
5. (Optional) visualize patterns

## Data Model

Each step should be represented as:

Step = {
    "thought": str,
    "action": str,
    "result": str,
}

## Core Metrics

### Repetition Rate

repetition_rate = repeated_actions / total_actions

### Loop Detection

Detect cycles of repeated failed actions.

### Action Diversity

diversity = unique_actions / total_actions

### Failure Persistence

failure_retry_rate = retries_after_failure / total_failures

### Thought–Action Divergence

Compare semantic similarity between thoughts and actions using embeddings.

### Semantic Drift Over Time

Measure distance between consecutive thoughts.

### Command Validity Approximation

Estimate syntactic validity of commands.

### Exploration vs Exploitation

exploration_ratio = new_actions / total_actions

## Optional Advanced Metrics

### Obsession Score

obsession_score = max_consecutive_repeats / total_occurrences

### Cognitive Inertia

inertia = average_repeats_after_failure

### Dead-End Detection

Detect stagnation zones.

## Implementation Plan

### Step 1 — Parser

Extract thoughts, actions, results.

### Step 2 — Structuring

Build Step objects.

### Step 3 — Metrics Engine

Compute metrics.

### Step 4 — Output

Export JSON summary.

### Step 5 — Visualization (optional)

Graphs and timelines.

## Technical Stack

- Python 3.10+
- numpy / pandas
- sentence-transformers or Ollama embeddings
- matplotlib (optional)

## Expected Outcome

LLMs show:

- no persistent state
- local coherence
- poor planning
- repetitive patterns

## Research Perspective

This tool exposes limitations rather than fixing them.

> LLMs generate meaning but struggle with constrained symbolic systems.

## Next Steps

- analyze multiple logs
- compare models
- integrate into thesis
