# Phoebe Project Memory

## Current State

Phoebe is an AI safety agent adapted from an AT Protocol Trust & Safety architecture into a **Sandbox Arena** — a red-teaming marketplace where adversarial researchers attack target AI models and Phoebe acts as the centralized safety judge.

### Branch: `claude/explain-codebase-mle8k2lr9c27lof0-S2PFT`

Latest commit: `1d57ec4` — Add SentientAGI/crypto-agent-safe-function-calling to safety RL pipeline

## What's Done

### Core Infrastructure
- **Arena Server** (`src/arena/server.py`) — Starlette ASGI with x402 payment verification, EIP-191 signatures, rate limiting
- **ClickHouse Store** (`src/arena/store.py`) — 7 tables: bounties, submissions, evaluations, attack_history, leaderboard, payment_log, tns_breach_log
- **Scorer** (`src/arena/scorer.py`) — α×attack + β×novelty + γ×coverage - δ×duplicate formula with oracle + payout integration
- **Config** (`src/config.py`) — Full configuration: ClickHouse, model, x402, arena, safety, oracle, escrow, DPO training

### Safety Classification
- **SafetyClassifier** (`src/safety/classifier.py`) — LLM-as-judge with GA Guard policy, single + batch classification
- **GA Guard Policy** (`src/osprey/policy.py`) — 7 categories (PII/IP, Illicit, Hate, Sexual, Prompt Security, Violence, Misinfo) with block/allow rules and compliance anchors
- **Taxonomy** (`src/arena/taxonomy.py`) — SafetyCategory enum with GA Guard + legacy categories, LEGACY_TO_GA_GUARD mapping

### Blockchain / x402
- **Wallet** (`src/x402/wallet.py`) — EVM wallet with EIP-191 signing, USDC contract addresses per chain
- **x402 Client** (`src/x402/client.py`) — HTTP 402 payment flow
- **Oracle** (`src/safety/oracle.py`) — OraclePublisher for PhoebeOracle contract (publish, query, is_unsafe, get_severity, get_history)
- **Oracle Indexer** (`src/safety/oracle_indexer.py`) — Event polling for ResultPublished
- **Payout Proof** (`src/safety/payout_proof.py`) — EIP-191 signed payout authorization + lightweight evaluation receipts
- **Escrow Contract** (`contracts/PhoebeEscrow.sol`) — USDC escrow with signature-verified claims
- **Oracle Contract** (`contracts/PhoebeOracle.sol`) — On-chain evaluation result storage

### TEE / ERC-8004 (Feature Branch — Self-Contained)
- **TEE Classifier** (`src/safety/tee_classifier.py`) — Phala TEE proxy wrapping SafetyClassifier
- **TEE Config** (`src/safety/tee_config.py`) — Separate TEEConfig from main Config
- **ERC-8004** (`src/safety/erc8004.py`) — Attestation NFT minting

### Agent Modes
- **System Prompts** (`src/agent/prompt.py`) — Judge, Admin, Red Team modes with `build_system_prompt(mode)`
- **Tool Registry** — admin (14 tools), attack (4 tools), oracle (4 tools), policy (4 tools), plus content/domain/ip/safety/url tools

### T&S Dashboard
- **Dashboard Backend** (`src/ui/dashboard.py`) — 8 API routes: classify, breaches, stats, generate, execute, analyze-logs, bounties
- **Dashboard Frontend** (`src/ui/dashboard.html`) — Dark-themed SPA with prompt checker, category cards, attack generator (8 evasion techniques), log analyzer, file upload, CSV import/export

### Safety RL Training
- **DPO Pipeline** (`src/safety/safety_rl_pipeline.py`) — 5 dataset adapters: R-Judge, PKU-SafeRLHF, HarmBench, BeaverTails, CrAI-SafeFuncCall
- **DPO Trainer** (`src/safety/dpo_trainer.py`) — TRL DPOTrainer wrapper with configurable hyperparameters

### CLI
- **main.py** — Commands: arena, chat, admin, redteam, pull-safety-data, train-dpo, generate-synthetic, eval

## What's NOT Done (Proposed Only)

### Ozone Enforcement Layer
- `src/ozone/ozone.py` is a stub (NotImplementedError)
- Proposed: SYNC/ASYNC/QUARANTINE enforcement modes
- Proposed: `enforcement_log` ClickHouse table
- Proposed: `rule_performance_metrics` table with false-positive tracking
- Proposed: Automated rollback at >2% false-positive rate
- Proposed: `/api/ozone/evaluate` HTTP endpoint

### DAO Safety Module (CrAI-SafeFuncCall Integration)
- Proposed: Three-stage verification (Format Validation → Execution Simulation → Semantic Alignment)
- Proposed: Risk tiers T1-T4 with tier-specific enforcement modes
- Proposed: FunctionCallValidator for DAO function call schema checking
- Proposed: SimulationRunner for dry-run execution
- Proposed: Multi-sig hold for T4 critical operations
- CrAI-SafeFuncCall DPO adapter is implemented (training data ready)

### AI Output Monitoring
- Proposed: Real-time output monitoring via HTTP proxy / SDK hook / streaming tap
- Proposed: "Live Monitor" dashboard tab
- Proposed: Rule performance heatmap
- Proposed: On-chain enforcement attestations

## Eval Harness (Separate Branch)
- Located in `src/eval/` on the eval-harness branch
- 10 Indeed policy mappings to GA Guard
- Seed dataset (30 hand-curated examples)
- Synthetic violation generator
- Metrics: F1, precision, recall, confusion matrix
- 29 tests in `tests/test_eval_harness.py`

## Key Design Decisions
1. **x402 over Bittensor** — Stable USDC payments with Phoebe as centralized judge instead of decentralized subnet
2. **TEE as optional feature** — Moved off main code path into self-contained module
3. **GA Guard as canonical taxonomy** — All 7 categories with compliance anchors, legacy categories mapped via LEGACY_TO_GA_GUARD
4. **LLM-as-judge pattern** — SafetyClassifier uses Claude to evaluate content against policy rules (same pattern as Phoebe's core agent)
5. **ClickHouse for persistence** — ReplacingMergeTree with FINAL for upsert semantics; ngramDistance for novelty detection
