# Sara — Cryptographic AI Safety Platform

Sara is an AI-powered trust and safety platform with cryptographic audit trails.
Standalone project (formerly forked from haileyok/phoebe — see [ATTRIBUTION.md](ATTRIBUTION.md)).

## Two-Agent Architecture

```
┌─────────────────────────────────┐   ┌──────────────────────────────────┐
│            SARA                 │   │             SHEILA               │
│   Safety Classifier · Monitor   │   │   Red Team Judge · Attacker      │
│                                 │   │                                  │
│  POST /classify                 │   │  judge mode  → SheilaVerdict     │
│  SkillFile / Sara-in-a-Box      │◄──►  redteam mode → RedTeamReport    │
│  Osprey rules enforcement       │   │  admin mode  → bounty mgmt       │
│  Ozone STOP/ALERT/LOG           │   │                                  │
│  ClickHouse audit trail         │   │  ECDSA P-384 EvaluationAttestation│
│  SHA3-256 L1 commitments        │   │  ERC8004 on-chain records        │
└─────────────────────────────────┘   └──────────────────────────────────┘
         Sara faces outward                   Sheila faces inward
     (classifies real interactions)      (probes Sara + other AI systems)
                    │                              │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │   ZK Audit Trail      │
                    │  L1: SHA3-256 commit  │
                    │  L2: ECDSA P-384 att  │
                    │  L3: SP1 STARK (Ph5)  │
                    └──────────────────────┘
```

**Critical rule**: Sara code never imports Sheila internals directly. All Sara↔Sheila
communication goes through `agents/sheila/api.py`. This boundary makes Phase 5 TEE
extraction mechanical — swapping direct calls for A2A HTTP requires no changes to Sara's call sites.

## Module Status (PRD v2)

| Module | Location | Status | Description |
|--------|----------|--------|-------------|
| Sara Safety Classifier | `src/safety/classifier.py` | Stable | LLM-as-judge, 12 DAO categories |
| Sara Monitor | `src/safety/monitor.py` | Stable | Rule set v0.1, Sheila forwarding |
| Osprey Rule Engine | `src/safety/osprey_client.py` | Stable | Kafka adapter, SML rules, fallback |
| Ozone Enforcement | `src/ozone/ozone.py` | Stable | SYNC/ASYNC/QUARANTINE + rollback |
| Sheila Judge API | `agents/sheila/api.py` | Stable | Public interface, network-transparent |
| Sheila Judge (local) | `agents/sheila/judge.py` | Stable | LLM-backed, stub fallback |
| Sheila Red Team | `agents/sheila/red_team.py` | Stable | HMAC-signed probe IDs |
| Sheila A2A Client | `agents/sheila/a2a_client.py` | Phase 5 | TEE stub, raises NotImplementedError |
| DPO Dataset | `src/data/dpo_dataset.py` | Stable | 18 base pairs, CoT SFT, ATLAS labels |
| DPO Loader | `src/data/dpo_loader.py` | Stable | Augmentation, train/val/test splits |
| ZK Audit L1 | `src/crypto/commitment.py` | Stable | SHA3-256 commit/reveal, CAT-02 chain |
| ZK Audit L2 | `src/crypto/attestation.py` | Stable | ECDSA P-384, ephemeral key warning |
| Attesting Agent | `src/crypto/attesting_agent.py` | Stable | L1+L2 wire-up, ERC8004 publish |
| Arena Store | `src/arena/store.py` | Stable | ClickHouse, attestations table, GDPR |
| ERC8004 Publisher | `src/safety/erc8004.py` | Stable | On-chain attestation tokens |
| Sara-in-a-Box | `src/sarabox/` | Stable | SkillFile, taxonomy, API |
| Osprey UI | `src/osprey_ui/` | Stable | Rule management, monitoring |
| DPO Training Script | `scripts/train_dpo.py` | Ready | Qwen2.5-7B-Instruct base, LoRA |

## Quick Start

### Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager
- Python 3.12+
- ClickHouse (optional — in-memory mode available)

### Installation

```bash
git clone https://github.com/smfang/sara.git
cd sara
uv sync --frozen
```

### Configuration

```bash
cp .env.example .env
# Edit .env — minimum required:
# MOONSHOT_API_KEY or ANTHROPIC_API_KEY
```

Minimum `.env` contents:

```env
# Required (default provider: kimi / Moonshot)
MOONSHOT_API_KEY="your-moonshot-key"
MODEL_API=kimi
MODEL_NAME=kimi-k2

# Or use Anthropic
# ANTHROPIC_API_KEY="your-anthropic-key"
# MODEL_API=anthropic
# MODEL_NAME=claude-sonnet-4-5-20250929

# Arena server
ARENA_HOST=0.0.0.0
ARENA_PORT=8080

# Sheila integration (Phase 5 TEE — optional)
# SHEILA_A2A_URL=http://enclave:8080
# SHEILA_ATTESTATION_KEY_PEM=<ECDSA P-384 PEM>

# ClickHouse (optional)
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=8123
```

### Running the Arena Server

```bash
uv run main.py serve
```

Then open `http://localhost:8080/researcher` for the researcher portal.

### Running Tests

```bash
uv run python -m pytest tests/ -v --tb=short
```

### DPO Training (Sheila Judge)

Generate the dataset and train:

```bash
# Generate DPO dataset (18 base × 2 augmentation = 36 pairs)
uv run python -m src.data.dpo_loader

# Train Sheila judge model
uv run python scripts/train_dpo.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --train_data data/dpo/train.json \
  --val_data data/dpo/val.json \
  --output_dir models/sheila-judge-dpo
# Estimated cost: $10–30/run via Replicate or Modal
```

## How It Works

Sara uses a model API as its reasoning backer. The agent writes and executes Typescript code in a sandboxed Deno runtime to interact with its tools.

```
┌─────────────────────────────────────────────────────────┐
│                       Model API                         │
├─────────────────────────────────────────────────────────┤
│              Tool Execution (Deno Sandbox)              │
├──────────┬───────────┬──────────────┬───────────────────┤
│  Osprey  │ ClickHouse│    Ozone     │  Investigation    │
│  (Rules) │ (Queries) │ (Moderation) │ (Domain/IP/WHOIS) │
└──────────┴───────────┴──────────────┴───────────────────┘
```

| Limit | Value |
|-------|-------|
| Max code size | 50,000 characters |
| Max tool calls per execution | 25 |
| Max output size | 1 MB |
| Execution timeout | 60 seconds |
| V8 heap memory | 256 MB |

## Regulatory Anchors

- **EU AI Act** — Art. 9, 12–15, 17, 43 (transparency, accuracy, human oversight)
- **NIST AI RMF** — Govern, Map, Measure, Manage
- **MITRE ATLAS** — Adversarial ML threat matrix (v2025-10)
- **GDPR** — CAT-05: raw prompts never stored, only SHA3-256 hash
- **DORA** — Digital Operational Resilience Act (financial sector)
- **ERC8004** — On-chain attestation standard for TEE-verified classifications

## Licence

MIT + Apache 2.0 (dual licence — see [LICENSE](LICENSE) and [ATTRIBUTION.md](ATTRIBUTION.md))
