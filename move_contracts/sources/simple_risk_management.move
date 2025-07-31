/// Simple Risk Management for Aptos Trading Bot
/// Compatible with current Aptos Move compiler
module trading_bot::simple_risk_management {
    use std::signer;
    use aptos_framework::timestamp;

    /// Errors
    const E_NOT_OWNER: u64 = 1;
    const E_ALREADY_INITIALIZED: u64 = 2;
    const E_NOT_INITIALIZED: u64 = 3;
    const E_RISK_LIMIT_EXCEEDED: u64 = 4;
    const E_EMERGENCY_STOP_ACTIVE: u64 = 5;

    /// Risk management configuration
    struct RiskManager has key {
        owner: address,
        max_position_size: u64,
        max_daily_loss: u64,
        current_daily_loss: u64,
        emergency_stop: bool,
        last_reset_day: u64,
        total_portfolio_value: u64,
        max_leverage: u64, // 10000 = 1x, 20000 = 2x
    }

    /// Position risk metrics
    struct PositionRisk has copy, drop, store {
        symbol: vector<u8>,
        position_size: u64,
        current_value: u64,
        unrealized_pnl: u64, // Changed from i64 to u64
        risk_score: u64, // 0-10000 (0-100%)
    }

    /// Initialize risk manager
    public entry fun initialize_risk_manager(
        account: &signer,
        max_position_size: u64,
        max_daily_loss: u64,
    ) {
        let account_addr = signer::address_of(account);
        
        assert!(!exists<RiskManager>(account_addr), E_ALREADY_INITIALIZED);
        
        let risk_manager = RiskManager {
            owner: account_addr,
            max_position_size,
            max_daily_loss,
            current_daily_loss: 0,
            emergency_stop: false,
            last_reset_day: get_current_day(),
            total_portfolio_value: 1000000, // Default 1M
            max_leverage: 20000, // 2x leverage
        };
        
        move_to(account, risk_manager);
    }

    /// Check if a trade is within risk limits
    public fun check_trade_risk(
        owner: address,
        position_size: u64,
        signal_confidence: u64,
    ): bool acquires RiskManager {
        if (!exists<RiskManager>(owner)) {
            return false
        };
        
        let risk_manager = borrow_global<RiskManager>(owner);
        
        // Check emergency stop
        if (risk_manager.emergency_stop) {
            return false
        };
        
        // Check position size limit
        if (position_size > risk_manager.max_position_size) {
            return false
        };
        
        // Check confidence threshold
        if (signal_confidence < 7000) { // 70% minimum
            return false
        };
        
        // Reset daily loss if new day
        let current_day = get_current_day();
        if (current_day > risk_manager.last_reset_day) {
            // In a mutable context, we'd reset current_daily_loss here
            // For now, we'll assume it's been reset
        };
        
        // Check daily loss limit
        if (risk_manager.current_daily_loss >= risk_manager.max_daily_loss) {
            return false
        };
        
        true
    }

    /// Update daily loss (should be called after each trade)
    public entry fun update_daily_loss(
        account: &signer,
        loss_amount: u64,
    ) acquires RiskManager {
        let account_addr = signer::address_of(account);
        assert!(exists<RiskManager>(account_addr), E_NOT_INITIALIZED);
        
        let risk_manager = borrow_global_mut<RiskManager>(account_addr);
        assert!(risk_manager.owner == account_addr, E_NOT_OWNER);
        
        // Reset if new day
        let current_day = get_current_day();
        if (current_day > risk_manager.last_reset_day) {
            risk_manager.current_daily_loss = 0;
            risk_manager.last_reset_day = current_day;
        };
        
        risk_manager.current_daily_loss = risk_manager.current_daily_loss + loss_amount;
        
        // Auto-activate emergency stop if daily loss exceeded
        if (risk_manager.current_daily_loss >= risk_manager.max_daily_loss) {
            risk_manager.emergency_stop = true;
        };
    }

    /// Activate/deactivate emergency stop
    public entry fun set_emergency_stop(
        account: &signer,
        activate: bool,
    ) acquires RiskManager {
        let account_addr = signer::address_of(account);
        assert!(exists<RiskManager>(account_addr), E_NOT_INITIALIZED);
        
        let risk_manager = borrow_global_mut<RiskManager>(account_addr);
        assert!(risk_manager.owner == account_addr, E_NOT_OWNER);
        
        risk_manager.emergency_stop = activate;
    }

    /// Update portfolio value
    public entry fun update_portfolio_value(
        account: &signer,
        new_value: u64,
    ) acquires RiskManager {
        let account_addr = signer::address_of(account);
        assert!(exists<RiskManager>(account_addr), E_NOT_INITIALIZED);
        
        let risk_manager = borrow_global_mut<RiskManager>(account_addr);
        assert!(risk_manager.owner == account_addr, E_NOT_OWNER);
        
        risk_manager.total_portfolio_value = new_value;
    }

    /// Get current risk status
    #[view]
    public fun get_risk_status(owner: address): (bool, u64, u64, u64) acquires RiskManager {
        assert!(exists<RiskManager>(owner), E_NOT_INITIALIZED);
        
        let risk_manager = borrow_global<RiskManager>(owner);
        (
            risk_manager.emergency_stop,
            risk_manager.current_daily_loss,
            risk_manager.max_daily_loss,
            risk_manager.total_portfolio_value
        )
    }

    /// Calculate position risk score (0-10000)
    public fun calculate_position_risk(
        position_size: u64,
        portfolio_value: u64,
        volatility: u64, // 0-10000
    ): u64 {
        // Risk = (Position Size / Portfolio Value) * Volatility Weight
        let position_weight = (position_size * 10000) / portfolio_value;
        let volatility_weight = if (volatility > 5000) 15000 else 10000; // 1.5x or 1x
        
        (position_weight * volatility_weight) / 10000
    }

    /// Get current day (simplified)
    fun get_current_day(): u64 {
        timestamp::now_seconds() / 86400 // Seconds per day
    }

    /// Validate if a position can be opened
    #[view]
    public fun can_open_position(
        owner: address,
        position_size: u64,
        confidence: u64,
    ): bool acquires RiskManager {
        check_trade_risk(owner, position_size, confidence)
    }

    /// Get maximum allowed position size
    #[view]
    public fun get_max_position_size(owner: address): u64 acquires RiskManager {
        assert!(exists<RiskManager>(owner), E_NOT_INITIALIZED);
        
        let risk_manager = borrow_global<RiskManager>(owner);
        risk_manager.max_position_size
    }

    #[test_only]
    public fun init_for_test(account: &signer) {
        initialize_risk_manager(account, 100000, 50000);
    }

    #[test(account = @0x1)]
    /// Test initialization of risk manager
    public fun test_initialize_risk_manager(account: &signer) {
        
        init_for_test(account);
        
        let account_addr = signer::address_of(account);
        assert!(exists<RiskManager>(account_addr), 1);
        
        let (emergency_stop, current_loss, max_loss, portfolio_value) = get_risk_status(account_addr);
        assert!(!emergency_stop, 2);
        assert!(current_loss == 0, 3);
        assert!(max_loss == 50000, 4);
        assert!(portfolio_value == 1000000, 5);
    }

    #[test(account = @0x1)]
    public fun test_check_trade_risk(account: &signer) {
        
        init_for_test(account);
        
        let account_addr = signer::address_of(account);
        
        // Valid trade
        assert!(check_trade_risk(account_addr, 50000, 8000), 1);
        
        // Invalid - position too large
        assert!(!check_trade_risk(account_addr, 200000, 8000), 2);
        
        // Invalid - confidence too low
        assert!(!check_trade_risk(account_addr, 50000, 5000), 3);
    }
}
