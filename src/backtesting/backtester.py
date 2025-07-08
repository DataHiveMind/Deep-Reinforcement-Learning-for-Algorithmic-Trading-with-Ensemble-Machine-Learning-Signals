import backtrader as bt
import pandas as pd
from typing import Any, Dict

class AgentStrategy(bt.Strategy):
    def __init__(self, agent, obs_columns, ml_columns=None):
        self.agent = agent
        self.obs_columns = obs_columns
        self.ml_columns = ml_columns
        self.actions = []
        self.rewards = []
        self.infos = []
        self.trade_history = []

    def next(self):
        obs = [getattr(self.datas[0], col)[0] for col in self.obs_columns]
        if self.ml_columns:
            ml_obs = [getattr(self.datas[0], col)[0] for col in self.ml_columns]
            obs = obs + ml_obs
        obs = pd.Series(obs).values.astype('float32')
        action = self.agent.predict(obs)
        # Action: 0=Buy, 1=Sell, 2=Hold
        if action == 0 and not self.position:
            self.buy()
            self.trade_history.append({'step': len(self.actions), 'action': 'buy', 'price': self.datas[0].close[0]})
        elif action == 1 and self.position:
            self.sell()
            self.trade_history.append({'step': len(self.actions), 'action': 'sell', 'price': self.datas[0].close[0]})
        # Hold does nothing
        self.actions.append(action)
        self.rewards.append(0)
        self.infos.append({})

class Backtester:
    """
    Backtester using Backtrader to run a trained agent on historical data.
    """
    def __init__(self, agent, data: pd.DataFrame, obs_columns, ml_columns=None, cash=10000.0):
        self.agent = agent
        self.data = data
        self.obs_columns = obs_columns
        self.ml_columns = ml_columns
        self.cash = cash
        self.history = []

    def run(self, render: bool = False) -> Dict[str, Any]:
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(self.cash)
        datafeed = bt.feeds.PandasData(dataname=self.data)
        cerebro.adddata(datafeed)
        strategy = cerebro.addstrategy(AgentStrategy, agent=self.agent, obs_columns=self.obs_columns, ml_columns=self.ml_columns)
        results = cerebro.run()
        strat = results[0]
        portfolio_value = cerebro.broker.getvalue()
        return {
            'final_portfolio_value': portfolio_value,
            'trade_history': strat.trade_history,
            'actions': strat.actions
        }

    def run_buy_and_hold(self) -> Dict[str, Any]:
        """Backtest a buy-and-hold strategy."""
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(self.cash)
        datafeed = bt.feeds.PandasData()
        datafeed.dataname = self.data
        cerebro.adddata(datafeed)
        class BuyAndHold(bt.Strategy):
            def __init__(self):
                self.bought = False
            def next(self):
                if not self.bought:
                    self.buy()
                    self.bought = True
        cerebro.addstrategy(BuyAndHold)
        cerebro.run()
        return {'final_portfolio_value': cerebro.broker.getvalue()}

    def run_random_agent(self) -> Dict[str, Any]:
        """Backtest a random agent."""
        import random
        class RandomAgent:
            def predict(self, obs):
                return random.choice([0, 1, 2])
        random_agent = RandomAgent()
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(self.cash)
        datafeed = bt.feeds.PandasData()
        datafeed.dataname = self.data
        cerebro.adddata(datafeed)
        cerebro.addstrategy(AgentStrategy, agent=random_agent, obs_columns=self.obs_columns, ml_columns=self.ml_columns)
        results = cerebro.run()
        strat = results[0]
        return {'final_portfolio_value': cerebro.broker.getvalue(), 'actions': strat.actions}

    def run_sell_and_hold(self) -> Dict[str, Any]:
        """Backtest a sell-and-hold (short) strategy."""
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(self.cash)
        datafeed = bt.feeds.PandasData()
        datafeed.dataname = self.data
        cerebro.adddata(datafeed)
        class SellAndHold(bt.Strategy):
            def __init__(self):
                self.sold = False
            def next(self):
                if not self.sold:
                    self.sell()
                    self.sold = True
        cerebro.addstrategy(SellAndHold)
        cerebro.run()
        return {'final_portfolio_value': cerebro.broker.getvalue()}

    def run_threshold_strategy(self, buy_threshold: float, sell_threshold: float) -> Dict[str, Any]:
        """Backtest a simple threshold-based strategy on the first obs_column."""
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(self.cash)
        datafeed = bt.feeds.PandasData()
        datafeed.dataname = self.data
        cerebro.adddata(datafeed)
        obs_col = self.obs_columns[0]
        class ThresholdStrategy(bt.Strategy):
            def next(self):
                val = getattr(self.datas[0], obs_col)[0]
                if val > buy_threshold and not self.position:
                    self.buy()
                elif val < sell_threshold and self.position:
                    self.sell()
        cerebro.addstrategy(ThresholdStrategy)
        cerebro.run()
        return {'final_portfolio_value': cerebro.broker.getvalue()}

    def run_walk_forward(self, window_size: int = 100) -> Dict[str, Any]:
        """Backtest using walk-forward (rolling window) validation."""
        n = len(self.data)
        results = []
        for start in range(0, n - window_size, window_size):
            end = start + window_size
            window_data = self.data.iloc[start:end]
            cerebro = bt.Cerebro()
            cerebro.broker.set_cash(self.cash)
            datafeed = bt.feeds.PandasData()
            datafeed.dataname = window_data
            cerebro.adddata(datafeed)
            cerebro.addstrategy(AgentStrategy, agent=self.agent, obs_columns=self.obs_columns, ml_columns=self.ml_columns)
            cerebro.run()
            results.append(cerebro.broker.getvalue())
        return {'walk_forward_portfolio_values': results}
