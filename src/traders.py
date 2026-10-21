"""Trader portfolios and the three market decision rules."""

import numpy as np


class Trader:
    """Cash, holdings, activity status, and marked-to-market wealth history."""

    def __init__(self, trader_type, cash, assets):
        self.type = trader_type
        self.cash = max(cash, 0)
        self.assets = max(assets, 0)
        self.wealth_history = [cash + assets * 100]
        self.active = True

    def update_wealth(self, current_price):
        self.wealth_history.append(self.cash + self.assets * current_price)

    def can_buy(self, price, qty):
        return self.cash >= price * qty and qty > 0

    def can_sell(self, qty):
        return self.assets >= qty and qty > 0


class NoiseTrader(Trader):
    """Place randomized orders around the current market price."""

    def __init__(self, cash, assets):
        super().__init__('noise', cash, assets)

    def make_decision(self, asset, rn=0.05):
        if not self.active:
            return None, None, None
        rand_val = np.random.rand()
        if rand_val > 0.6 and self.cash > 0:
            price = asset.current_price * (1 + rn * np.random.rand())
            max_qty = int(self.cash / price)
            if max_qty <= 0:
                return None, None, None
            qty = min(max_qty, self.assets if hasattr(self, 'assets') else max_qty)
            qty = max(1, int(qty * 0.1))
            if self.can_buy(price, qty):
                return 'buy', price, qty
        elif rand_val < 0.4 and self.assets > 0:
            price = asset.current_price * (1 - rn * np.random.rand())
            qty = min(self.assets, max(1, int(self.assets * 0.1)))
            if self.can_sell(qty):
                return 'sell', price, qty
        return None, None, None


class ChartistTrader(Trader):
    """Trade around a moving average with volatility-dependent buy markups."""

    def __init__(self, cash, assets, window=20, alpha=0.1):
        super().__init__('chartist', cash, assets)
        self.window = window
        self.alpha = alpha

    def make_decision(self, asset):
        if not self.active or len(asset.price_history) < self.window:
            return None, None, None
        prices = asset.price_history[-self.window:]
        sma = np.mean(prices)
        current_price = asset.current_price
        if current_price > sma and self.cash > 0:
            volatility = np.std(prices) / sma
            correction = 0.02 * (1 + volatility * 10) * np.random.rand()
            price = current_price * (1 + correction)
            max_qty = int(self.cash * self.alpha / price)
            if max_qty <= 0:
                return None, None, None
            qty = min(max_qty, self.assets if hasattr(self, 'assets') else max_qty)
            qty = max(1, qty)
            if self.can_buy(price, qty):
                return 'buy', price, qty
        elif self.assets > 0:
            price = current_price * (1 - 0.02 * np.random.rand())
            qty = min(self.assets, max(1, int(self.assets * self.alpha)))
            if self.can_sell(qty):
                return 'sell', price, qty
        return None, None, None


class FundamentalistTrader(Trader):
    """Choose target holdings from dividend-adjusted excess returns."""

    def __init__(self, cash, assets, gamma=2, rf=0.0004):
        super().__init__('fundamentalist', cash, assets)
        self.gamma = gamma
        self.rf = rf

    def make_decision(self, asset):
        if not self.active or len(asset.price_history) < 2:
            return None, None, None
        Pt = asset.current_price
        Pt_1 = asset.price_history[-2]
        dt_1 = asset.dividend_history[-2]
        rt_d = (asset.dividend_history[-1] - dt_1) / dt_1 if dt_1 != 0 else 0
        r_excess = (Pt/Pt_1 - 1) + (dt_1 * (1 + rt_d)/Pt_1) - self.rf
        if abs(r_excess) > 0.05:
            xt = max(0.1, min(0.9, r_excess / (self.gamma * 0.1)))
            desired_assets = int((xt * (self.cash + self.assets * Pt)) / Pt)
            if desired_assets > self.assets and self.cash > 0:
                qty = min(desired_assets - self.assets, int(self.cash / (Pt * 1.01)))
                qty = max(1, qty)
                price = Pt * 1.01
                if self.can_buy(price, qty):
                    return 'buy', price, qty
            elif desired_assets < self.assets and self.assets > 0:
                qty = min(self.assets - desired_assets, self.assets)
                qty = max(1, qty)
                price = Pt * 0.99
                if self.can_sell(qty):
                    return 'sell', price, qty
        return None, None, None
