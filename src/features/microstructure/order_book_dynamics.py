"""
Order Book Dynamics Features.

This module extracts high-frequency features from L1/L2 Order Book data.
Key metrics included:
- Order Book Imbalance (OBI)
- Weighted Mid Price (WMP)
- Book Slope/Pressure

References:
    - Cartea, Á., Jaimungal, S., & Penalva, J. (2015). 
      Algorithmic and High-Frequency Trading.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple

class OrderBookDynamics:
    """
    Calculates advanced order book features for microstructure analysis.
    """
    
    @staticmethod
    def calculate_obi(
        bid_vol: Union[pd.Series, np.ndarray], 
        ask_vol: Union[pd.Series, np.ndarray]
    ) -> Union[pd.Series, np.ndarray]:
        """
        Calculates Order Book Imbalance (OBI).
        
        Formula: (Bid_Vol - Ask_Vol) / (Bid_Vol + Ask_Vol)
        Range: [-1, 1]
        
        A positive OBI indicates buying pressure (bids > asks).
        """
        total_vol = bid_vol + ask_vol
        
        # Handle division by zero
        obi = np.divide(
            (bid_vol - ask_vol), 
            total_vol, 
            out=np.zeros_like(total_vol, dtype=float), 
            where=total_vol!=0
        )
        
        if isinstance(bid_vol, pd.Series):
            return pd.Series(obi, index=bid_vol.index, name="obi")
        return obi

    @staticmethod
    def calculate_wmp(
        bid_px: Union[pd.Series, np.ndarray],
        bid_vol: Union[pd.Series, np.ndarray],
        ask_px: Union[pd.Series, np.ndarray],
        ask_vol: Union[pd.Series, np.ndarray]
    ) -> Union[pd.Series, np.ndarray]:
        """
        Calculates Volume-Weighted Mid Price (WMP).
        
        WMP gives more weight to the side with LESS volume, as price 
        is more likely to move towards the side with less resistance.
        
        Formula: (BidPx * AskVol + AskPx * BidVol) / (BidVol + AskVol)
        """
        total_vol = bid_vol + ask_vol
        
        wmp = np.divide(
            (bid_px * ask_vol + ask_px * bid_vol),
            total_vol,
            out=np.zeros_like(total_vol, dtype=float),
            where=total_vol!=0
        )
        
        # Fallback to simple mid price if volume is 0
        simple_mid = (bid_px + ask_px) / 2
        mask = (total_vol == 0)
        
        if isinstance(wmp, np.ndarray):
            wmp[mask] = simple_mid[mask]
        else:
            wmp = wmp.fillna(simple_mid)
            
        return wmp

    @staticmethod
    def calculate_mid_price_divergence(
        mid_price: Union[pd.Series, np.ndarray],
        wmp: Union[pd.Series, np.ndarray]
    ) -> Union[pd.Series, np.ndarray]:
        """
        Calculates divergence between Simple Mid Price and Weighted Mid Price.
        Signal: If WMP > Mid Price, potential upward pressure.
        """
        return wmp - mid_price