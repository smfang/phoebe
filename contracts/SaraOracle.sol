// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title SaraOracle
 * @notice On-chain oracle for Sara safety evaluation results.
 *
 * Stores structured classification results that are readable by both
 * smart contracts (via view functions) and AI agents (via events).
 *
 * Only authorized Sara instances (whitelisted addresses or TEE-verified
 * signers) can publish results. Any contract or off-chain agent can read.
 */
contract SaraOracle {
    // ---------------------------------------------------------------
    // Types
    // ---------------------------------------------------------------

    struct EvaluationResult {
        bytes32 promptHash;
        uint8   category;
        uint8   severity;      // 1-5
        bool    unsafe;
        uint64  timestamp;
        bytes32 enclaveHash;   // TEE enclave measurement (0x0 if not TEE)
        bytes   attestation;   // raw attestation or signature bytes
    }

    // ---------------------------------------------------------------
    // Events — AI agents subscribe to these via websocket / RPC
    // ---------------------------------------------------------------

    event ResultPublished(
        bytes32 indexed evaluationId,
        bytes32 indexed promptHash,
        uint8   category,
        uint8   severity,
        bool    unsafe
    );

    event PublisherAuthorized(address indexed publisher);
    event PublisherRevoked(address indexed publisher);

    // ---------------------------------------------------------------
    // State
    // ---------------------------------------------------------------

    address public owner;

    /// evaluationId => EvaluationResult
    mapping(bytes32 => EvaluationResult) private _results;

    /// promptHash => category => evaluationId (latest per prompt+category)
    mapping(bytes32 => mapping(uint8 => bytes32)) private _latestByPrompt;

    /// promptHash => all evaluationIds
    mapping(bytes32 => bytes32[]) private _historyByPrompt;

    /// authorized publishers (Sara instances)
    mapping(address => bool) public authorizedPublishers;

    /// total results published
    uint256 public totalResults;

    // ---------------------------------------------------------------
    // Modifiers
    // ---------------------------------------------------------------

    modifier onlyOwner() {
        require(msg.sender == owner, "SaraOracle: not owner");
        _;
    }

    modifier onlyAuthorized() {
        require(authorizedPublishers[msg.sender], "SaraOracle: not authorized");
        _;
    }

    // ---------------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------------

    constructor() {
        owner = msg.sender;
        authorizedPublishers[msg.sender] = true;
    }

    // ---------------------------------------------------------------
    // Admin — publisher management
    // ---------------------------------------------------------------

    function authorizePublisher(address publisher) external onlyOwner {
        authorizedPublishers[publisher] = true;
        emit PublisherAuthorized(publisher);
    }

    function revokePublisher(address publisher) external onlyOwner {
        authorizedPublishers[publisher] = false;
        emit PublisherRevoked(publisher);
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "SaraOracle: zero address");
        owner = newOwner;
    }

    // ---------------------------------------------------------------
    // Write — publish evaluation results
    // ---------------------------------------------------------------

    /**
     * @notice Publish a safety evaluation result on-chain.
     * @param promptHash    SHA-256 of the evaluated prompt
     * @param category      Safety category index (maps to GA Guard policy)
     * @param severity      Severity level 1-5
     * @param unsafe        Whether the output was classified as unsafe
     * @param attestation   TEE attestation quote or evaluator signature
     * @return evaluationId Unique identifier for this result
     */
    function publishResult(
        bytes32 promptHash,
        uint8   category,
        uint8   severity,
        bool    unsafe,
        bytes calldata attestation
    ) external onlyAuthorized returns (bytes32 evaluationId) {
        require(severity >= 1 && severity <= 5, "SaraOracle: severity must be 1-5");

        evaluationId = keccak256(
            abi.encodePacked(promptHash, category, severity, unsafe, block.timestamp, totalResults)
        );

        _results[evaluationId] = EvaluationResult({
            promptHash:  promptHash,
            category:    category,
            severity:    severity,
            unsafe:      unsafe,
            timestamp:   uint64(block.timestamp),
            enclaveHash: bytes32(0),
            attestation: attestation
        });

        _latestByPrompt[promptHash][category] = evaluationId;
        _historyByPrompt[promptHash].push(evaluationId);
        totalResults++;

        emit ResultPublished(evaluationId, promptHash, category, severity, unsafe);
    }

    /**
     * @notice Publish with TEE enclave hash for verifiable execution proof.
     */
    function publishResultWithTEE(
        bytes32 promptHash,
        uint8   category,
        uint8   severity,
        bool    unsafe,
        bytes32 enclaveHash,
        bytes calldata attestation
    ) external onlyAuthorized returns (bytes32 evaluationId) {
        require(severity >= 1 && severity <= 5, "SaraOracle: severity must be 1-5");
        require(enclaveHash != bytes32(0), "SaraOracle: empty enclave hash");

        evaluationId = keccak256(
            abi.encodePacked(promptHash, category, severity, unsafe, enclaveHash, block.timestamp, totalResults)
        );

        _results[evaluationId] = EvaluationResult({
            promptHash:  promptHash,
            category:    category,
            severity:    severity,
            unsafe:      unsafe,
            timestamp:   uint64(block.timestamp),
            enclaveHash: enclaveHash,
            attestation: attestation
        });

        _latestByPrompt[promptHash][category] = evaluationId;
        _historyByPrompt[promptHash].push(evaluationId);
        totalResults++;

        emit ResultPublished(evaluationId, promptHash, category, severity, unsafe);
    }

    // ---------------------------------------------------------------
    // Read — for smart contracts
    // ---------------------------------------------------------------

    /**
     * @notice Get full evaluation result by ID.
     */
    function getResult(bytes32 evaluationId)
        external view returns (EvaluationResult memory)
    {
        EvaluationResult memory r = _results[evaluationId];
        require(r.timestamp != 0, "SaraOracle: result not found");
        return r;
    }

    /**
     * @notice Check if a prompt was classified as unsafe for a given category.
     * @dev    Returns the latest result for the (promptHash, category) pair.
     *         Returns false if no result exists (safe by default).
     */
    function isUnsafe(bytes32 promptHash, uint8 category)
        external view returns (bool)
    {
        bytes32 evalId = _latestByPrompt[promptHash][category];
        if (evalId == bytes32(0)) return false;
        return _results[evalId].unsafe;
    }

    /**
     * @notice Get the severity of the latest result for a prompt+category.
     * @return severity 0 if no result, 1-5 otherwise
     */
    function getSeverity(bytes32 promptHash, uint8 category)
        external view returns (uint8)
    {
        bytes32 evalId = _latestByPrompt[promptHash][category];
        if (evalId == bytes32(0)) return 0;
        return _results[evalId].severity;
    }

    /**
     * @notice Get the latest evaluationId for a prompt+category.
     */
    function getLatestEvaluationId(bytes32 promptHash, uint8 category)
        external view returns (bytes32)
    {
        return _latestByPrompt[promptHash][category];
    }

    /**
     * @notice Get all evaluationIds for a prompt (across all categories).
     */
    function getHistory(bytes32 promptHash)
        external view returns (bytes32[] memory)
    {
        return _historyByPrompt[promptHash];
    }
}
