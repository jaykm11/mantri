"""
Technical Indicators Calculator

Calculates basic technical indicators from historical stock price data.
Supports various moving averages, momentum indicators, and volatility measures.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from .mongodb_provider import MongoDBProvider

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculator for basic technical indicators
    """
    
    def __init__(self, mongodb_provider: MongoDBProvider):
        """
        Initialize with MongoDB provider
        
        Args:
            mongodb_provider: MongoDBProvider instance
        """
        self.db_provider = mongodb_provider
        
        # Technical indicators configuration
        self.indicator_configs = {
            "sma": [10, 20, 50, 200],  # Simple Moving Average periods
            "ema": [10, 20, 50, 200],  # Exponential Moving Average periods
            "wma": [10, 20],           # Weighted Moving Average periods
            "dema": [10, 20],          # Double Exponential Moving Average periods
            "tema": [10, 20],          # Triple Exponential Moving Average periods
            "rsi": [14, 20],           # Relative Strength Index periods
            "adx": [14, 20],           # Average Directional Index periods
            "williams": [14, 20],      # Williams %R periods
            "stdev": [10, 20],         # Standard Deviation periods
            "momentum": [10, 20],      # Momentum periods
            "roc": [10, 20]            # Rate of Change periods
        }
    
    def calculate_sma(self, prices: pd.Series, period: int) -> Optional[float]:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return None
        return float(prices.tail(period).mean())
    
    def calculate_ema(self, prices: pd.Series, period: int) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return None
        return float(prices.ewm(span=period, adjust=False).mean().iloc[-1])
    
    def calculate_wma(self, prices: pd.Series, period: int) -> Optional[float]:
        """Calculate Weighted Moving Average"""
        if len(prices) < period:
            return None
        
        weights = np.arange(1, period + 1)
        recent_prices = prices.tail(period)
        return float(np.average(recent_prices, weights=weights))
    
    def calculate_dema(self, prices: pd.Series, period: int) -> Optional[float]:
        """Calculate Double Exponential Moving Average"""
        if len(prices) < period * 2:
            return None
        
        ema1 = prices.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        return float((2 * ema1.iloc[-1]) - ema2.iloc[-1])
    
    def calculate_tema(self, prices: pd.Series, period: int) -> Optional[float]:
        """Calculate Triple Exponential Moving Average"""
        if len(prices) < period * 3:
            return None
        
        ema1 = prices.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return float((3 * ema1.iloc[-1]) - (3 * ema2.iloc[-1]) + ema3.iloc[-1])
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return None
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate Average Directional Index"""
        if len(df) < period + 1:
            return None
        
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # True Range calculation
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            
            # Smoothed averages
            atr = tr.ewm(alpha=1/period, adjust=False).mean()
            plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
            minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
            
            # ADX calculation
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(alpha=1/period, adjust=False).mean()
            
            return float(adx.iloc[-1])
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return None
    
    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate Williams %R"""
        if len(df) < period:
            return None
        
        try:
            recent_df = df.tail(period)
            highest_high = recent_df['high'].max()
            lowest_low = recent_df['low'].min()
            current_close = df['close'].iloc[-1]
            
            williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
            return float(williams_r)
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {e}")
            return None
    
    def calculate_standard_deviation(self, prices: pd.Series, period: int) -> Optional[float]:
        """Calculate Standard Deviation"""
        if len(prices) < period:
            return None
        return float(prices.tail(period).std())
    
    def calculate_momentum(self, prices: pd.Series, period: int) -> Optional[float]:
        """Calculate Momentum (price change over period)"""
        if len(prices) < period + 1:
            return None
        
        current_price = prices.iloc[-1]
        period_ago_price = prices.iloc[-(period + 1)]
        return float(current_price - period_ago_price)
    
    def calculate_rate_of_change(self, prices: pd.Series, period: int) -> Optional[float]:
        """Calculate Rate of Change (percentage change over period)"""
        if len(prices) < period + 1:
            return None
        
        current_price = prices.iloc[-1]
        period_ago_price = prices.iloc[-(period + 1)]
        
        if period_ago_price == 0:
            return None
        
        roc = ((current_price - period_ago_price) / period_ago_price) * 100
        return float(roc)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate Average True Range"""
        if len(df) < period + 1:
            return None
        
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # True Range calculation
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Average True Range
            atr = tr.ewm(alpha=1/period, adjust=False).mean()
            return float(atr.iloc[-1])
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return None
    
    def calculate_indicators_for_symbol(self, symbol: str, date: str,
                                      lookback_days: int = 252) -> Dict:
        """
        Calculate all technical indicators for a symbol on a specific date
        
        Args:
            symbol: Stock symbol
            date: Date in YYYY-MM-DD format
            lookback_days: Number of days to look back for calculations
            
        Returns:
            Dictionary with calculated indicators
        """
        logger.info(f"Calculating technical indicators for {symbol} on {date}")
        
        # Get historical price data
        df = self.db_provider.get_historical_prices(symbol, date, lookback_days)
        if df.empty:
            logger.warning(f"No price data available for {symbol} on {date}")
            return {}
        
        # Filter to only include data up to the target date
        df = df[df['ds'] <= date]
        
        if df.empty:
            logger.warning(f"No price data found for {symbol} up to {date}")
            return {}
        
        # Initialize indicators dictionary
        indicators = {
            "symbol": symbol,
            "ds": date,
            "calculated_at": pd.Timestamp.now().isoformat()
        }
        
        try:
            close_prices = df['close'].dropna()
            
            # Calculate Moving Averages
            for period in self.indicator_configs["sma"]:
                sma_val = self.calculate_sma(close_prices, period)
                if sma_val is not None:
                    indicators[f"sma_{period}"] = round(sma_val, 4)
            
            for period in self.indicator_configs["ema"]:
                ema_val = self.calculate_ema(close_prices, period)
                if ema_val is not None:
                    indicators[f"ema_{period}"] = round(ema_val, 4)
            
            for period in self.indicator_configs["wma"]:
                wma_val = self.calculate_wma(close_prices, period)
                if wma_val is not None:
                    indicators[f"wma_{period}"] = round(wma_val, 4)
            
            for period in self.indicator_configs["dema"]:
                dema_val = self.calculate_dema(close_prices, period)
                if dema_val is not None:
                    indicators[f"dema_{period}"] = round(dema_val, 4)
            
            for period in self.indicator_configs["tema"]:
                tema_val = self.calculate_tema(close_prices, period)
                if tema_val is not None:
                    indicators[f"tema_{period}"] = round(tema_val, 4)
            
            # Calculate Momentum Indicators
            for period in self.indicator_configs["rsi"]:
                rsi_val = self.calculate_rsi(close_prices, period)
                if rsi_val is not None:
                    indicators[f"rsi_{period}"] = round(rsi_val, 4)
            
            # Calculate ADX (requires OHLC data)
            for period in self.indicator_configs["adx"]:
                adx_val = self.calculate_adx(df, period)
                if adx_val is not None:
                    indicators[f"adx_{period}"] = round(adx_val, 4)
            
            # Calculate Williams %R (requires OHLC data)
            for period in self.indicator_configs["williams"]:
                williams_val = self.calculate_williams_r(df, period)
                if williams_val is not None:
                    indicators[f"williams_r_{period}"] = round(williams_val, 4)
            
            # Calculate Volatility Indicators
            for period in self.indicator_configs["stdev"]:
                std_val = self.calculate_standard_deviation(close_prices, period)
                if std_val is not None:
                    indicators[f"stdev_{period}"] = round(std_val, 4)
            
            # Calculate Additional Momentum Indicators
            for period in self.indicator_configs["momentum"]:
                momentum_val = self.calculate_momentum(close_prices, period)
                if momentum_val is not None:
                    indicators[f"momentum_{period}"] = round(momentum_val, 4)
            
            for period in self.indicator_configs["roc"]:
                roc_val = self.calculate_rate_of_change(close_prices, period)
                if roc_val is not None:
                    indicators[f"roc_{period}"] = round(roc_val, 4)
            
            # Calculate ATR
            atr_14 = self.calculate_atr(df, 14)
            if atr_14 is not None:
                indicators["atr_14"] = round(atr_14, 4)
            
            atr_20 = self.calculate_atr(df, 20)
            if atr_20 is not None:
                indicators["atr_20"] = round(atr_20, 4)
            
            logger.info(f"Calculated {len(indicators)-3} indicators for {symbol} on {date}")
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return {}
    
    def calculate_and_save_indicators(self, symbol: str = None, date: str = None,
                                    lookback_days: int = 252) -> bool:
        """
        Calculate and save technical indicators for symbol(s) and date(s)
        
        Args:
            symbol: Stock symbol (if None, processes all symbols for date)
            date: Date in YYYY-MM-DD format (if None, uses today)
            lookback_days: Number of days to look back for calculations
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Set default date if not provided
            if date is None:
                date = pd.Timestamp.now().strftime("%Y-%m-%d")
            
            # Get symbols to process
            if symbol:
                symbols = [symbol]
            else:
                symbols = self.db_provider.get_symbols_for_date(date)
            
            if not symbols:
                logger.warning(f"No symbols found for processing on {date}")
                return False
            
            logger.info(f"Processing {len(symbols)} symbols for {date}")
            
            # Calculate indicators for each symbol
            all_indicators = []
            for sym in symbols:
                try:
                    indicators = self.calculate_indicators_for_symbol(
                        sym, date, lookback_days
                    )
                    if indicators and len(indicators) > 3:  # More than just symbol, ds, calculated_at
                        all_indicators.append(indicators)
                except Exception as e:
                    logger.error(f"Error processing {sym}: {e}")
                    continue
            
            # Save indicators
            if all_indicators:
                success = self.db_provider.save_indicators(
                    all_indicators, 
                    self.db_provider.technical_indicators_collection_name
                )
                if success:
                    logger.info(f"Successfully saved technical indicators for "
                              f"{len(all_indicators)} symbols on {date}")
                    return True
                else:
                    logger.error("Failed to save technical indicators")
                    return False
            else:
                logger.warning(f"No indicators calculated for any symbols on {date}")
                return False
                
        except Exception as e:
            logger.error(f"Error in calculate_and_save_indicators: {e}")
            return False
    
    def get_indicators_summary(self, symbol: str, start_date: str = None,
                             end_date: str = None) -> pd.DataFrame:
        """
        Get a summary of calculated indicators for a symbol
        
        Args:
            symbol: Stock symbol
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            DataFrame with indicator summary
        """
        try:
            df = self.db_provider.get_indicators(
                symbol, start_date, end_date,
                self.db_provider.technical_indicators_collection_name
            )
            
            if df.empty:
                logger.info(f"No technical indicators found for {symbol}")
                return pd.DataFrame()
            
            # Remove non-indicator columns for summary
            exclude_cols = ['_id', 'symbol', 'ds', 'calculated_at']
            indicator_cols = [col for col in df.columns if col not in exclude_cols]
            
            summary = {
                'symbol': symbol,
                'total_records': len(df),
                'date_range': f"{df['ds'].min()} to {df['ds'].max()}",
                'indicators_calculated': len(indicator_cols),
                'indicator_names': indicator_cols
            }
            
            return pd.DataFrame([summary])
            
        except Exception as e:
            logger.error(f"Error getting indicators summary for {symbol}: {e}")
            return pd.DataFrame()
