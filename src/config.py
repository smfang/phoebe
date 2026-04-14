from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    # clickhouse config
    clickhouse_host: str = "localhost"
    """host for the clickhouse server"""
    clickhouse_port: int = 8123
    """port for the clickhouse server"""
    clickhouse_user: str = "default"
    """username for the clickhouse server"""
    clickhouse_password: str = "clickhouse"
    """password for the clickhouse server"""
    clickhouse_database: str = "default"
    """default database for the clickhouse server"""

    # model config — used for Phoebe's own LLM reasoning + safety classifier
    model_api: Literal["anthropic", "openai", "openapi"] = "anthropic"
    """the model api to use. must be one of `anthropic`, `openai`, or `openapi`"""
    model_name: str = "claude-sonnet-4-5-20250929"
    """the model to use with the given api"""
    model_api_key: str = ""
    """the model api key"""
    model_endpoint: str = ""
    """for openapi model apis, the endpoint to use"""

    # x402 payment config
    x402_wallet_private_key: str = ""
    """private key for signing x402 USDC payments (EVM or Solana)"""
    x402_wallet_address: str = ""
    """wallet address for x402 payments"""
    x402_chain: str = "base"
    """blockchain to use for x402 payments (base, solana, etc.)"""
    x402_facilitator_url: str = ""
    """x402 facilitator URL for payment settlement"""
    x402_max_auto_pay: float = 1.0
    """maximum USDC amount to auto-pay per x402 request"""

    # arena config
    arena_host: str = "0.0.0.0"
    """host for the arena HTTP server"""
    arena_port: int = 8080
    """port for the arena HTTP server"""
    arena_submission_fee: float = 0.01
    """USDC fee per attack submission (anti-spam)"""
    arena_scoring_alpha: float = 0.4
    """scoring weight for attack success"""
    arena_scoring_beta: float = 0.3
    """scoring weight for novelty"""
    arena_scoring_gamma: float = 0.2
    """scoring weight for category coverage"""
    arena_scoring_delta: float = 0.1
    """scoring weight for duplicate penalty"""
    arena_payout_rate: float = 1.0
    """score-to-USDC multiplier for payouts"""
    arena_wallet: str = "arena.sandbox.eth"
    """arena's wallet address for receiving bounty funds and submission fees"""
    arena_dev_mode: bool = True
    """run arena in dev mode (accept DevWallet HMAC signatures, skip real EVM verification)"""

    # spending limits
    x402_spending_limit: float = 100.0
    """cumulative USDC spending limit for x402 client"""

    # safety classifier config
    safety_classifier_model: str = "claude-sonnet-4-5-20250929"
    """model to use for the LLM-as-judge safety classifier"""
    safety_classifier_endpoint: str = "https://api.anthropic.com"
    """API endpoint for the safety classifier"""

    # oracle config — on-chain evaluation result publishing
    oracle_contract_address: str = ""
    """PhoebeOracle contract address (empty = oracle disabled)"""
    oracle_chain: str = "base"
    """blockchain where the oracle contract is deployed"""
    oracle_rpc_url: str = ""
    """JSON-RPC URL for oracle contract interactions"""
    oracle_relayer_url: str = ""
    """HTTP relayer URL for oracle publishing (alternative to direct RPC)"""
    oracle_publisher_address: str = ""
    """address authorized to publish results to the oracle"""
    oracle_private_key: str = ""
    """private key for signing oracle transactions (if direct mode)"""

    # escrow config — two-phase x402 payout with proof
    escrow_contract_address: str = ""
    """PhoebeEscrow contract address (empty = escrow disabled)"""
    escrow_enabled: bool = False
    """enable two-phase payout proofs (requires wallet + oracle)"""

    # safety RL / DPO training config
    dpo_base_model: str = ""
    """HuggingFace model ID to fine-tune with DPO (empty = use default)"""
    dpo_data_dir: str = "data/safety_rl"
    """directory for the cached DPO preference dataset"""
    dpo_output_dir: str = "models/safety_dpo"
    """directory to save the fine-tuned DPO model"""
    dpo_max_per_source: int = 10_000
    """max rows to pull from each dataset source"""

    # domain modules — pluggable safety verification per problem domain
    dao_enabled: bool = False
    """register the DAO domain module at startup (three-stage function-call verification)"""
    dao_simulation_rpc_url: str = "http://localhost:8545"
    """JSON-RPC endpoint for the DAO simulation backend (Anvil/Foundry/Tenderly)"""
    dao_alignment_model_name: str = ""
    """override model for the DAO alignment judge; empty = reuse safety_classifier_model"""
    dao_alignment_endpoint: str = ""
    """override endpoint for the DAO alignment judge; empty = reuse safety_classifier_endpoint"""
    dao_scammer_addresses: str = ""
    """comma-separated allow-list of known-scammer addresses for the DAO validator"""

    model_config = SettingsConfigDict(env_file=".env")


CONFIG = Config()
