from datetime import datetime, timedelta
import math

import pandas as pd
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy
import logging
logger = logging.getLogger(__name__)
import talib.abstract as ta


class AlmgrenChrissStrategy(IStrategy):
    """
    Almgren-Chriss optimal execution strategy.
    """
    timeframe = "15m"
    stoploss = -0.10
    minimal_roi = {"0": 0.02}
    process_only_new_candles = True
    startup_candle_count = 30
    can_short = True
    position_adjustment_enable = True

    twap_num_slices = 10            # number of slices
    twap_interval_minutes = 1       # interval between slices
    vol_window = 96                 # rolling window to use 
    factor_lambda = 0.01            # risk-aversion - the one dial you tune globally
    eta_volume_fraction = 0.01      # Temporary impact coef
    gamma_volume_fraction = 0.1     # Permanent impact coef

    kappa_default = 0.6             # default kappa 
    kappa_max = 5.0                 # max kappa 

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.kappa: dict[str, float] = {}

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe)
        self._ac_update_kappa_for_pair(dataframe, metadata["pair"])
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe.loc[
            (dataframe["rsi"] < 45) & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1

        # Short entry
        dataframe.loc[
            (dataframe["rsi"] > 55) & (dataframe["volume"] > 0),
            "enter_short",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        return dataframe

    def ome_populate_exit_trend(self, trade: Trade, current_time: datetime) -> bool:
        """
        Exit trigger condition, checked directly here so it can drive a
        exit. Replace with your actual signal/profit/time logic.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(
            trade.pair, self.timeframe
        )

        if dataframe.empty:
            return False

        last_candle = dataframe.iloc[-1]
        rsi = last_candle["rsi"]

        if trade.is_short:
            return rsi < 45

        return rsi > 55

    def _ac_update_kappa_for_pair(self, dataframe: pd.DataFrame, pair: str) -> None:
        """
        Reference code: https://github.com/joshuapjacob/almgren-chriss-optimal-execution
        """
        if dataframe.empty  :
            self.kappa[pair] = self.kappa.get(pair, self.kappa_default)
            return

        window = dataframe.tail(self.vol_window)
        if len(window) < 5:
            self.kappa[pair] = self.kappa.get(pair, self.kappa_default)
            return

        sigma = window["close"].std()
        avg_spread = (window["high"] - window["low"]).mean()
        avg_volume = window["volume"].mean()

        candle_minutes = timeframe_to_minutes(self.timeframe)
        tau = self.twap_interval_minutes / candle_minutes
        eta = avg_spread / (self.eta_volume_fraction * avg_volume)
        gamma = avg_spread / (self.gamma_volume_fraction * avg_volume)
        eta_tilde = eta - 0.5 * gamma * tau

        if eta_tilde <= 0:
            logger.warning(
                "%s: eta_tilde <= 0 (eta=%.6g, gamma=%.6g, tau=%.4g); "
                "falling back to eta_tilde=eta",
                pair, eta, gamma, tau,
            )
            eta_tilde = eta
        sigma_tau = sigma * math.sqrt(tau)
        kappa_tilde_sq = (self.factor_lambda * sigma_tau ** 2) / eta_tilde
        acosh_arg = 0.5 * kappa_tilde_sq * tau ** 2 + 1.0
        raw_kappa = math.acosh(acosh_arg) / tau
        new_kappa = min(raw_kappa, self.kappa_max)
        self.kappa[pair] = new_kappa


    def _get_kappa(self, pair: str) -> float:
        return self.kappa.get(pair, self.kappa_default)

    def _ac_next_slice_fraction(self, remaining_slices: int, kappa: float) -> float:
        """
        Fraction of the *currently remaining* amount to trade in the next
        slice, given `remaining_slices` .
        kappa == 0 reduces exactly to 1/remaining_slices plain TWAP.
        """
        m = remaining_slices
        if m <= 1:
            return 1.0
        if kappa <= 1e-12:
            return 1.0 / m

        denominator = math.sinh(kappa * m)
        if denominator == 0:
            return 1.0 / m

        numerator = math.sinh(kappa * (m - 1))
        frac = 1.0 - (numerator / denominator)
        return min(max(frac, 0.0), 1.0)


    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                             proposed_stake: float, min_stake: float | None, max_stake: float,
                             leverage: float, entry_tag: str | None, side: str,
                             **kwargs) -> float:
        first_fraction = self._ac_next_slice_fraction(self.twap_num_slices, self._get_kappa(pair))
        return proposed_stake * first_fraction

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                               current_rate: float, current_profit: float,
                               min_stake: float | None, max_stake: float,
                               current_entry_rate: float, current_exit_rate: float,
                               current_entry_profit: float, current_exit_profit: float,
                               **kwargs
                               ) -> float | None | tuple[float | None, str | None]:

        if trade.has_open_orders:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        entry_slices_done = len(filled_entries)
        filled_exits = trade.select_filled_orders(trade.exit_side)
        exit_slices_done = len(filled_exits)

        already_exiting = exit_slices_done > 0

        if already_exiting or self.ome_populate_exit_trend(trade, current_time):
            return self._next_exit_slice(trade, current_time, filled_exits, exit_slices_done)

        if entry_slices_done < self.twap_num_slices:
            return self._next_entry_slice(trade, current_time, filled_entries, entry_slices_done, min_stake)

        return None

    def _next_entry_slice(self, trade: Trade, current_time: datetime,
                           filled_entries: list, slices_done: int, min_stake: float | None
                           ) -> float | None | tuple[float | None, str | None]:

        last_fill_time = filled_entries[-1].order_filled_utc if filled_entries else trade.open_date_utc
        next_slice_due_at = last_fill_time + timedelta(minutes=self.twap_interval_minutes)
        if current_time < next_slice_due_at:
            return None

        kappa = self._get_kappa(trade.pair)

        first_fraction = self._ac_next_slice_fraction(self.twap_num_slices, kappa)
        ac_total_stake = filled_entries[0].stake_amount_filled / first_fraction

        stake_already_filled = sum(o.stake_amount_filled for o in filled_entries)
        remaining_stake = ac_total_stake - stake_already_filled
        remaining_slices = self.twap_num_slices - slices_done

        if remaining_stake <= 0 or remaining_slices <= 0:
            return None

        fraction = self._ac_next_slice_fraction(remaining_slices, kappa)
        next_slice_stake = remaining_stake * fraction

        if next_slice_stake < (min_stake or 0):
            return None

        return next_slice_stake

    def _next_exit_slice(self, trade: Trade, current_time: datetime,
                          filled_exits: list, slices_done: int
                          ) -> float | None | tuple[float | None, str | None]:

        if slices_done >= self.twap_num_slices:
            return None

        last_fill_time = filled_exits[-1].order_filled_utc if filled_exits else current_time
        next_slice_due_at = last_fill_time + timedelta(minutes=self.twap_interval_minutes)
        if slices_done > 0 and current_time < next_slice_due_at:
            return None

        remaining_slices = self.twap_num_slices - slices_done

        if remaining_slices <= 1:
            return -trade.stake_amount
        fraction = self._ac_next_slice_fraction(remaining_slices, self._get_kappa(trade.pair))
        slice_stake = trade.stake_amount * fraction
        return -slice_stake