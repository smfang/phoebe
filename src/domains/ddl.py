"""DDL for tables shared across all domain modules."""

DOMAIN_DDL = [
    "CREATE DATABASE IF NOT EXISTS arena",
    """
    CREATE TABLE IF NOT EXISTS arena.domain_evaluations (
        request_id      String,
        domain          LowCardinality(String),
        module_version  String,
        tier            LowCardinality(String),
        mode            LowCardinality(String),
        action          LowCardinality(String),
        severity        UInt8,
        reason          String,
        stage_results   String,
        decided_at      Float64,
        elapsed_ms      Float64
    ) ENGINE = MergeTree()
      ORDER BY (domain, decided_at, request_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS arena.dao_t4_pending (
        request_id          String,
        chain_id            UInt32,
        contract_address    String,
        function_signature  String,
        caller_address      String,
        decoded_params      String,
        created_at          Float64,
        timelock_expires_at Float64,
        required_sigs       UInt8,
        collected_sigs      Array(String),
        status              LowCardinality(String) DEFAULT 'PENDING'
    ) ENGINE = ReplacingMergeTree(created_at)
      ORDER BY request_id
    """,
]
