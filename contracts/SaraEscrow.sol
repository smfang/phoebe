// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./SaraOracle.sol";

/**
 * @title SaraEscrow
 * @notice Two-phase x402 escrow: bounty funds are locked until a verified
 *         vulnerability proof is presented, then USDC is released to the
 *         red teamer.
 *
 * Flow:
 *   1. Bounty funder calls fund() to deposit USDC into the escrow.
 *   2. Red teamer submits attack → Sara evaluates → publishes result
 *      to SaraOracle → signs a payout authorization.
 *   3. Red teamer (or Sara on their behalf) calls claim() with the
 *      signed payout authorization. The contract verifies:
 *        a. The signature is from an authorized Sara signer.
 *        b. The corresponding oracle result exists and is unsafe.
 *        c. The payout amount <= bounty remaining.
 *   4. USDC is transferred to the red teamer.
 */
interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract SaraEscrow {
    // ---------------------------------------------------------------
    // Types
    // ---------------------------------------------------------------

    struct BountyEscrow {
        address funder;
        uint256 totalFunded;
        uint256 remaining;
        bool    active;
    }

    struct PayoutClaim {
        bytes32 bountyId;
        bytes32 submissionId;
        bytes32 evaluationId;   // reference to SaraOracle result
        address recipient;
        uint256 amount;
        uint256 timestamp;
    }

    // ---------------------------------------------------------------
    // Events
    // ---------------------------------------------------------------

    event BountyFunded(bytes32 indexed bountyId, address indexed funder, uint256 amount);
    event PayoutClaimed(bytes32 indexed bountyId, bytes32 indexed submissionId, address indexed recipient, uint256 amount);
    event BountyWithdrawn(bytes32 indexed bountyId, address indexed funder, uint256 amount);

    // ---------------------------------------------------------------
    // State
    // ---------------------------------------------------------------

    address public owner;
    IERC20  public usdc;
    SaraOracle public oracle;

    /// bountyId => escrow
    mapping(bytes32 => BountyEscrow) public escrows;

    /// authorized Sara signers (can sign payout authorizations)
    mapping(address => bool) public authorizedSigners;

    /// submissionId => already claimed (prevents double-claim)
    mapping(bytes32 => bool) public claimed;

    // ---------------------------------------------------------------
    // Modifiers
    // ---------------------------------------------------------------

    modifier onlyOwner() {
        require(msg.sender == owner, "SaraEscrow: not owner");
        _;
    }

    // ---------------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------------

    constructor(address _usdc, address _oracle) {
        owner = msg.sender;
        usdc = IERC20(_usdc);
        oracle = SaraOracle(_oracle);
        authorizedSigners[msg.sender] = true;
    }

    // ---------------------------------------------------------------
    // Admin
    // ---------------------------------------------------------------

    function authorizeSigner(address signer) external onlyOwner {
        authorizedSigners[signer] = true;
    }

    function revokeSigner(address signer) external onlyOwner {
        authorizedSigners[signer] = false;
    }

    // ---------------------------------------------------------------
    // Fund — bounty creator deposits USDC
    // ---------------------------------------------------------------

    /**
     * @notice Deposit USDC into escrow for a bounty.
     * @dev    Caller must have approved this contract for `amount` USDC.
     */
    function fund(bytes32 bountyId, uint256 amount) external {
        require(amount > 0, "SaraEscrow: zero amount");
        require(usdc.transferFrom(msg.sender, address(this), amount), "SaraEscrow: transfer failed");

        BountyEscrow storage e = escrows[bountyId];
        if (!e.active) {
            e.funder = msg.sender;
            e.active = true;
        }
        e.totalFunded += amount;
        e.remaining += amount;

        emit BountyFunded(bountyId, msg.sender, amount);
    }

    // ---------------------------------------------------------------
    // Claim — red teamer claims payout with proof
    // ---------------------------------------------------------------

    /**
     * @notice Claim a payout for a verified vulnerability.
     * @param bountyId      The bounty being claimed against
     * @param submissionId  Unique submission ID (prevents double-claim)
     * @param evaluationId  SaraOracle evaluation ID proving the vuln
     * @param amount        USDC payout amount (in token decimals)
     * @param signature     EIP-191 signature from authorized Sara signer
     *                      over keccak256(bountyId, submissionId, evaluationId, recipient, amount)
     */
    function claim(
        bytes32 bountyId,
        bytes32 submissionId,
        bytes32 evaluationId,
        uint256 amount,
        bytes calldata signature
    ) external {
        require(!claimed[submissionId], "SaraEscrow: already claimed");

        BountyEscrow storage e = escrows[bountyId];
        require(e.active, "SaraEscrow: bounty not active");
        require(amount <= e.remaining, "SaraEscrow: insufficient funds");

        // Verify the oracle result exists and is unsafe
        SaraOracle.EvaluationResult memory result = oracle.getResult(evaluationId);
        require(result.unsafe, "SaraEscrow: result is not unsafe");

        // Verify the payout authorization signature
        bytes32 messageHash = keccak256(
            abi.encodePacked(bountyId, submissionId, evaluationId, msg.sender, amount)
        );
        bytes32 ethSignedHash = keccak256(
            abi.encodePacked("\x19Ethereum Signed Message:\n32", messageHash)
        );
        address signer = _recoverSigner(ethSignedHash, signature);
        require(authorizedSigners[signer], "SaraEscrow: invalid signer");

        // Execute payout
        claimed[submissionId] = true;
        e.remaining -= amount;

        require(usdc.transfer(msg.sender, amount), "SaraEscrow: payout transfer failed");

        emit PayoutClaimed(bountyId, submissionId, msg.sender, amount);
    }

    // ---------------------------------------------------------------
    // Withdraw — funder reclaims unused funds
    // ---------------------------------------------------------------

    function withdraw(bytes32 bountyId) external {
        BountyEscrow storage e = escrows[bountyId];
        require(msg.sender == e.funder, "SaraEscrow: not funder");
        require(e.remaining > 0, "SaraEscrow: nothing to withdraw");

        uint256 amount = e.remaining;
        e.remaining = 0;
        e.active = false;

        require(usdc.transfer(msg.sender, amount), "SaraEscrow: withdraw failed");

        emit BountyWithdrawn(bountyId, msg.sender, amount);
    }

    // ---------------------------------------------------------------
    // Internal — ECDSA recovery
    // ---------------------------------------------------------------

    function _recoverSigner(bytes32 hash, bytes memory sig)
        internal pure returns (address)
    {
        require(sig.length == 65, "SaraEscrow: invalid sig length");

        bytes32 r;
        bytes32 s;
        uint8 v;

        assembly {
            r := mload(add(sig, 32))
            s := mload(add(sig, 64))
            v := byte(0, mload(add(sig, 96)))
        }

        if (v < 27) v += 27;
        require(v == 27 || v == 28, "SaraEscrow: invalid v");

        return ecrecover(hash, v, r, s);
    }
}
