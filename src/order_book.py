"""Persistent limit-order storage and midpoint trade matching."""

from bisect import bisect_right
from operator import itemgetter

import numpy as np


class OrderBook:
    """Match bids and asks while retaining unmatched orders across trading days."""

    def __init__(self):
        self.buy_orders = []
        self.sell_orders = []

    def add_order(self, order_type, price, quantity, trader_id):
        if quantity <= 0:
            return
        if order_type == 'buy':
            # Insert after equal prices, matching the old stable descending sort.
            position = bisect_right(self.buy_orders, -price, key=lambda order: -order[0])
            self.buy_orders.insert(position, (price, quantity, trader_id))
        else:
            position = bisect_right(self.sell_orders, price, key=itemgetter(0))
            self.sell_orders.insert(position, (price, quantity, trader_id))

    def match_orders(self, traders, asset, *, on_trade=None, on_price=None):
        executed = []
        traders_dict = {id(trader): trader for trader in traders}
        while self.buy_orders and self.sell_orders and self.buy_orders[0][0] >= self.sell_orders[0][0]:
            buy_price, buy_qty, buyer_id = self.buy_orders[0]
            sell_price, sell_qty, seller_id = self.sell_orders[0]
            buyer = traders_dict.get(buyer_id)
            seller = traders_dict.get(seller_id)
            if not buyer or not seller:
                break
            if buyer.cash < buy_price * min(buy_qty, sell_qty):
                new_qty = int(buyer.cash / buy_price)
                if new_qty <= 0:
                    self.buy_orders.pop(0)
                    continue
                self.buy_orders[0] = (buy_price, new_qty, buyer_id)
                buy_qty = new_qty
            if seller.assets < min(buy_qty, sell_qty):
                new_qty = seller.assets
                if new_qty <= 0:
                    self.sell_orders.pop(0)
                    continue
                self.sell_orders[0] = (sell_price, new_qty, seller_id)
                sell_qty = new_qty
            exec_price = (buy_price + sell_price) / 2
            exec_qty = min(buy_qty, sell_qty)
            if exec_qty <= 0:
                break
            if buyer.cash >= exec_price * exec_qty and seller.assets >= exec_qty:
                if on_trade is not None:
                    before = {'buyer_cash': buyer.cash, 'buyer_assets': buyer.assets,
                              'seller_cash': seller.cash, 'seller_assets': seller.assets}
                executed.append((exec_price, exec_qty, buyer, seller))
                buyer.cash -= exec_price * exec_qty
                buyer.assets += exec_qty
                seller.cash += exec_price * exec_qty
                seller.assets -= exec_qty
                if on_trade is not None:
                    on_trade(exec_price, exec_qty, buyer, seller, before)
            else:
                break
            if buy_qty == exec_qty:
                self.buy_orders.pop(0)
            else:
                self.buy_orders[0] = (buy_price, buy_qty - exec_qty, buyer_id)
            if sell_qty == exec_qty:
                self.sell_orders.pop(0)
            else:
                self.sell_orders[0] = (sell_price, sell_qty - exec_qty, seller_id)
        if executed:
            total_qty = sum(qty for _, qty, _, _ in executed)
            if total_qty > 0:
                liquidity_factor = 1 / (1 + total_qty / 1000)
                volume_weighted_price = sum(price * qty for price, qty, _, _ in executed) / total_qty
                new_price = volume_weighted_price * (1 + 0.05 * liquidity_factor * np.random.randn())
                accepted = asset.update_price(max(new_price, 0.01))
                if on_price is not None:
                    on_price({'vwap': volume_weighted_price, 'liquidity_factor': liquidity_factor,
                              'proposed_price_before_floor': new_price,
                              'proposed_price': max(new_price, 0.01), 'accepted': accepted})
        return executed
