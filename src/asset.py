"""Asset prices, configurable dividends, and optional circuit-breaker state."""

import numpy as np


class Asset:
    """Single traded asset.

    circuit_breaker=False is the notebook's cell-4 Asset: every price update is accepted.
    circuit_breaker=True is its cell-42 Asset: a move of more than halt_threshold against last_check_price is
    rejected and halts trading. last_check_price is only refreshed by check_halt_status(), so without calling
    it the halt flag, once set, stays set (see SCENARIOS['evolution']).
    """
    def __init__(self, initial_price=100, circuit_breaker=False, *, log=print, event=None,
                 dividend_mode='stochastic'):
        self.log = log
        self.event = event
        self.price_history = [initial_price]
        self.dividend_history = [0.016]
        self.dividend_mode = dividend_mode
        self.current_price = initial_price
        self.circuit_breaker = circuit_breaker
        self.trading_halted = False
        self.halt_threshold = 0.20
        self.halt_duration = 5
        self.halt_remaining = 0
        self.last_check_price = initial_price

    def update_price(self, new_price):
        if self.circuit_breaker and not self.trading_halted:
            price_change = abs(new_price / self.last_check_price - 1)
            if price_change > self.halt_threshold:
                self.trading_halted = True
                self.halt_remaining = self.halt_duration
                if self.event is not None:
                    self.event('circuit_breaker_trigger', proposed_price=new_price,
                               reference_price=self.last_check_price, retained_price=self.current_price,
                               signed_return=new_price / self.last_check_price - 1,
                               absolute_return=price_change, threshold=self.halt_threshold,
                               duration=self.halt_duration)
                self.log(f"\nCircuit Breaker: Trading halted for {self.halt_duration} days (change {price_change:.2%} > {self.halt_threshold:.2%})")
                return False

        self.current_price = new_price
        self.price_history.append(new_price)
        return True

    def update_dividend(self, rd=0.0008, sigma_d=0.0008):
        if np.random.rand() < 0.01:
            shock = np.random.randn() * 0.005
        else:
            shock = 0
        new_div = self.dividend_history[-1] * (1 + rd + sigma_d * np.random.randn() + shock)
        # Retain every legacy draw, including the conditional shock, in control runs.
        if self.dividend_mode == 'constant':
            new_div = 0.016
        self.dividend_history.append(max(new_div, 0))
        return self.dividend_history[-1]

    def check_halt_status(self):
        if self.trading_halted:
            self.halt_remaining -= 1
            if self.halt_remaining <= 0:
                self.trading_halted = False
                self.last_check_price = self.current_price
                if self.event is not None:
                    self.event('circuit_breaker_resume', price=self.current_price,
                               remaining=self.halt_remaining)
                self.log("\nCircuit Breaker: Trading resumed")
        else:
            if len(self.price_history) > 0:
                self.last_check_price = self.price_history[-1]

        return self.trading_halted
