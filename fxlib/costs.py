"""Transaction cost model.

BlackBull's actual EUR/USD spread + commission were not available, so we
use a CONSERVATIVE PLACEHOLDER round-turn cost. Replace
DEFAULT_ROUND_TURN_PIPS with real figures (typical spread + commission per
lot converted to pips) before treating backtest results as realistic.
"""

from __future__ import annotations

import pandas as pd

PIP_SIZE_EURUSD = 0.0001

# PLACEHOLDER: ~1.0 pip round-turn (entry+exit combined). BlackBull Prime/ECN
# accounts often run tighter (e.g. ~0.1-0.4 pip spread + ~$3-7/lot commission,
# which is roughly another 0.3-0.7 pip on a standard lot) -- 1.0 pip
# round-turn is a deliberately cautious upper-end estimate until confirmed.
DEFAULT_ROUND_TURN_PIPS = 1.0


def cost_per_unit_position_change(
    price: pd.Series,
    round_turn_pips: float = DEFAULT_ROUND_TURN_PIPS,
    pip_size: float = PIP_SIZE_EURUSD,
) -> pd.Series:
    """Cost (as a fraction of price) charged per unit of |position change|.

    A round turn = one entry + one exit, so opening OR closing one unit of
    position costs half the round-turn cost.
    """
    return (round_turn_pips / 2.0) * pip_size / price
