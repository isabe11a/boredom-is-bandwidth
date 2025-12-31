# boredom-is-bandwidth
Treating boredom in reading instruction as a signal-processing problem.

## Overview

Boredom during reading instruction is usually framed as a learner problem:
lack of motivation, poor attention, or disengagement.

This project takes a different view.

We treat boredom as a **signal** that an instructional environment is transmitting
too little information and that the semantic bandwidth of the material has collapsed.

Rather than measuring boredom in students, this toolkit measures the
**informational properties of instructional text**.

## Core Idea

Learning systems (human or artificial) extract structure from variability.
When instructional materials are overly constrained, repetitive, or predictable,
they provide insufficient information for statistical learning.

This project operationalizes that idea by computing:

- **Semantic novelty**: how much new meaning is introduced over time
- **Redundancy / compressibility**: how predictable the surface form is
- **Contextual diversity**: how varied the usage contexts of words and ideas are

These metrics provide an approximate measure of the *semantic information rate*
of reading materials.

## What This Project Is (and Is Not)

**This project is:**
- A computational analysis toolkit for reading materials
- Grounded in statistical learning and information theory
- Useful for comparing curricula, texts, and instructional sequences

**This project is not:**
- A learner diagnostic tool
- A boredom detector
- A replacement for phonics, decoding instruction, or pedagogy

It evaluates **materials**, not **students**.

## Why This Matters

Connectionist and statistical accounts of reading suggest that:
- Some aspects of reading require explicit instruction
- Other aspects are learned automatically from structured exposure

Over-explicit instruction for non-dyslexic learners risks collapsing variability,
reducing discoverability, and lowering informational yield —
conditions under which boredom naturally arises.

This toolkit helps quantify those conditions.

## Metrics

### 1. Semantic Novelty
Measures how much new semantic information each page or passage adds
relative to previously seen material.

Computed using cosine distance in a reduced semantic space
(e.g., LSA / LSI).

### 2. Redundancy
Estimates surface-level repetition using compression ratios
and n-gram entropy.

Highly compressible text indicates high predictability.

### 3. Contextual Diversity
Measures how many distinct semantic neighborhoods a word appears in
across the corpus.

Low contextual diversity limits statistical learning opportunities.

## Typical Use Cases

- Compare decodable readers vs authentic texts
- Audit curriculum passages for over-constraint
- Track semantic information growth across a sequence of lessons
- Support differentiation without categorical labels

## Repository Structure


