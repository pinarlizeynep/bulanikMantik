# Fuzzy Logic-Based Computation Offloading in Fog Computing — Reproduction & Type-2 Extension

> Fuzzy Logic course term project. **Status: work in progress.**

## Student Information
- GitHub: [@pinarlizeynep](https://github.com/pinarlizeynep)
- Course: Bulanık Mantık (Fuzzy Logic), Fall 2026

## Original Paper
D. Soni and N. Kumar, "Fuzzy logic-based computation offloading technique in fog computing,"
*Concurrency and Computation: Practice and Experience*, vol. 36, no. 20, p. e8198, 2024,
doi: [10.1002/cpe.8198](https://doi.org/10.1002/cpe.8198).

## Problem Description
IoT devices generate workflow jobs that must be executed on one of three computing layers —
**edge**, **fog**, or **cloud**. The choice affects makespan, energy consumption and monetary cost,
and it must be made under uncertain, rapidly changing conditions (load, latency, bandwidth, deadlines).
The reference paper uses a Mamdani fuzzy inference system to make this offloading decision.

## Project Scope
1. **Reproduction** — Python re-implementation of the paper's 5-input Mamdani FIS
   (job list size, VM utilization, network latency, bandwidth, deadline → edge/fog/cloud)
   inside a lightweight workflow simulator modelled on the paper's FogWorkflowSim settings
   (MONTAGE-like workflows, MIN-MIN scheduling, 20–300 jobs).
2. **Proposed extension** — Interval Type-2 fuzzy offloading and a noise/robustness analysis,
   plus a Mamdani-vs-Sugeno / defuzzification ablation.

## Repository Structure
```
src/            fuzzy engine, environment model, workflow generator, scheduler, policies
data/           generated workflows and random seeds
experiments/    baseline/ and proposed/ experiment scripts
results/        figures/ and tables/
demo/           live demo
tests/          unit tests
docs/           project plan and paper analysis (Turkish)
report/         technical report (PDF)
presentation/   slides (PDF)
references/     bibliography
```

## Installation
```bash
git clone https://github.com/pinarlizeynep/bulanikMantik.git
cd bulanikMantik
py -3.12 -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Requirements
Python 3.12 — see [`requirements.txt`](requirements.txt) for pinned library versions.

## How to Run
_To be added._

## Reproduction Experiments
_To be added._

## Proposed Extension
_To be added._

## Experimental Results
_To be added._

## Limitations
_To be added._

## References
See [`references/README.md`](references/README.md).
