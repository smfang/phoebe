# Attribution

## Origin

Sara was originally forked from [haileyok/phoebe](https://github.com/haileyok/phoebe)
(also known as `osprey-agent`), an AI-powered trust and safety agent built by
[@haileyok](https://github.com/haileyok).

The original project provided the foundational architecture for:
- Tool-execution via sandboxed Deno runtime
- Osprey SML rule engine integration
- ClickHouse analytics backend
- Agent persona/mode configuration system (`AgentConfig`)

## Changes from origin

Sara v2.0 is a substantial independent evolution:

- **Two-agent architecture**: Sara (classifier) + Sheila (red team judge) separation
- **Cryptographic audit trail**: L1 SHA3-256 commitments, L2 ECDSA P-384 attestations
- **DPO pipeline**: Preference pair dataset, CoT SFT templates, LoRA training script
- **ZK foundation**: CAT-01/02/05 GDPR compliance, prev_hash chaining, ERC8004 on-chain records
- **Safety monitor**: Deterministic rule set (PromptInjection, AuthorityClaim, DataExfiltration, etc.)
- **Osprey integration**: Kafka adapter, SML rule files, Python fallback
- **Ozone enforcement**: SYNC/ASYNC/QUARANTINE modes with auto-rollback
- **Sara-in-a-Box**: Multi-org SkillFile system with DAO taxonomy
- **Arena**: Red teaming marketplace with x402 USDC payments

The internal agent name "Phoebe" was renamed to "Sheila" in v2.0 to reflect the
distinct role as a red team judge rather than a general-purpose assistant.

## Licence

The original codebase is MIT licensed. Sara v2.0 carries forward the MIT licence
for all components derived from the original, with Apache 2.0 applying to new
components added in this project.

See [LICENSE](LICENSE) for the full MIT licence text.

```
MIT License

Original work Copyright (c) haileyok
Derivative work Copyright (c) smfang and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```
