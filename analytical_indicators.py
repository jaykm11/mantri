"""
Analytical Indicators Calculator

Calculates advanced analytical indicators for stock analysis including
Bollinger Bands, MACD, Stochastic Oscillator, Ichimoku Cloud, and other
sophisticated technical analysis tools.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from .mongodb_provider import MongoDBProvider

logger = logging.getLogger(__name__)


class AnalyticalIndicators:
    """
    Calculator for advanced analytical indicators
    """
    
    def __init__(self, mongodb_provider: MongoDBProvider):
        """
        Initialize with MongoDB provider
        
        Args:
            mongodb_provider: MongoDBProvider instance
        """
        self.db_provider = mongodb_provider
        
        # Analytical indicators configuration
        self.indicator_configs = {
            "bollinger_bands": [20],           # Bollinger Bands periods
            "macd": [(12, 26, 9)],            # MACD (fast, slow, signal)
            "stochastic": [(14, 3, 3)],       # Stochastic (k_period, k_smooth, d_smooth)
            "cci": [20],                      # Commodity Channel Index periods
            "ichimoku": [(9, 26, 52)],        # Ichimoku (conversion, base, span_b)
            "parabolic_sar": [0.02],          # Parabolic SAR acceleration factor
            "fibonacci_retracements": [20],    # Fibonacci lookback periods
            "pivot_points": True,             # Pivot points calculation
            "support_resistance": [20]        # Support/Resistance lookback periods
        }
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20,
                                std_dev: float = 2.0) -> Dict:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, lower bands and bandwidth
        """
        if len(prices) < period:
            return {}
        
        try:
            # Calculate moving average (middle band)
            middle_band = prices.rolling(window=period).mean()
            
            # Calculate standard deviation
            std = prices.rolling(window=period).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            # Calculate bandwidth and %B
            bandwidth = ((upper_band - lower_band) / middle_band) * 100
            current_price = prices.iloc[-1]
            percent_b = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
            
            return {
                f"bb_upper_{period}": round(float(upper_band.iloc[-1]), 4),
                f"bb_middle_{period}": round(float(middle_band.iloc[-1]), 4),
                f"bb_lower_{period}": round(float(lower_band.iloc[-1]), 4),
                f"bb_bandwidth_{period}": round(float(bandwidth.iloc[-1]), 4),
                f"bb_percent_b_{period}": round(float(percent_b), 4)
            }
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {}
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26,
                      signal: int = 9) -> Dict:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        if len(prices) < slow + signal:
            return {}
        
        try:
            # Calculate EMAs
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return {
                f"macd_line_{fast}_{slow}_{signal}": round(float(macd_line.iloc[-1]), 4),
                f"macd_signal_{fast}_{slow}_{signal}": round(float(signal_line.iloc[-1]), 4),
                f"macd_histogram_{fast}_{slow}_{signal}": round(float(histogram.iloc[-1]), 4)
            }
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {}
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14,
                           k_smooth: int = 3, d_smooth: int = 3) -> Dict:
        """
        Calculate Stochastic Oscillator
        
        Args:
            df: DataFrame with high, low, close columns
            k_period: %K period
            k_smooth: %K smoothing period
            d_smooth: %D smoothing period
            
        Returns:
            Dictionary with %K and %D values
        """
        if len(df) < k_period + k_smooth + d_smooth:
            return {}
        
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate %K
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            
            # Smooth %K
            k_percent_smooth = k_percent.rolling(window=k_smooth).mean()
            
            # Calculate %D
            d_percent = k_percent_smooth.rolling(window=d_smooth).mean()
            
            return {
                f"stoch_k_{k_period}_{k_smooth}_{d_smooth}": round(float(k_percent_smooth.iloc[-1]), 4),
                f"stoch_d_{k_period}_{k_smooth}_{d_smooth}": round(float(d_percent.iloc[-1]), 4)
            }
            
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return {}
    
    def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> Dict:
        """
        Calculate Commodity Channel Index (CCI)
        
        Args:
            df: DataFrame with high, low, close columns
            period: CCI period
            
        Returns:
            Dictionary with CCI value
        """
        if len(df) < period:
            return {}
        
        try:
            # Calculate typical price
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Calculate moving average of typical price
            sma_tp = typical_price.rolling(window=period).mean()
            
            # Calculate mean deviation
            mean_dev = typical_price.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )
            
            # Calculate CCI
            cci = (typical_price - sma_tp) / (0.015 * mean_dev)
            
            return {
                f"cci_{period}": round(float(cci.iloc[-1]), 4)
            }
            
        except Exception as e:
            logger.error(f"Error calculating CCI: {e}")
            return {}
    
    def calculate_ichimoku_cloud(self, df: pd.DataFrame, conversion: int = 9,
                               base: int = 26, span_b: int = 52) -> Dict:
        """
        Calculate Ichimoku Cloud components
        
        Args:
            df: DataFrame with high, low, close columns
            conversion: Conversion line period
            base: Base line period
            span_b: Span B period
            
        Returns:
            Dictionary with Ichimoku components
        """
        if len(df) < span_b:
            return {}
        
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Conversion Line (Tenkan-sen)
            conv_high = high.rolling(window=conversion).max()
            conv_low = low.rolling(window=conversion).min()
            conversion_line = (conv_high + conv_low) / 2
            
            # Base Line (Kijun-sen)
            base_high = high.rolling(window=base).max()
            base_low = low.rolling(window=base).min()
            base_line = (base_high + base_low) / 2
            
            # Leading Span A (Senkou Span A)
            span_a = ((conversion_line + base_line) / 2).shift(base)
            
            # Leading Span B (Senkou Span B)
            span_b_high = high.rolling(window=span_b).max()
            span_b_low = low.rolling(window=span_b).min()
            span_b_val = ((span_b_high + span_b_low) / 2).shift(base)
            
            # Lagging Span (Chikou Span)
            lagging_span = close.shift(-base)
            
            return {
                f"ichimoku_conversion_{conversion}": round(float(conversion_line.iloc[-1]), 4),
                f"ichimoku_base_{base}": round(float(base_line.iloc[-1]), 4),
                f"ichimoku_span_a_{conversion}_{base}": round(float(span_a.iloc[-1]), 4),
                f"ichimoku_span_b_{span_b}": round(float(span_b_val.iloc[-1]), 4),
                f"ichimoku_lagging": round(float(lagging_span.iloc[-1-base]), 4)
            }
            
        except Exception as e:
            logger.error(f"Error calculating Ichimoku Cloud: {e}")
            return {}
    
    def calculate_parabolic_sar(self, df: pd.DataFrame, af: float = 0.02,
                              max_af: float = 0.2) -> Dict:
        """
        Calculate Parabolic SAR
        
        Args:
            df: DataFrame with high, low columns
            af: Acceleration factor
            max_af: Maximum acceleration factor
            
        Returns:
            Dictionary with Parabolic SAR value
        """
        if len(df) < 2:
            return {}
        
        try:
            high = df['high'].values
            low = df['low'].values
            
            # Initialize variables
            sar = np.zeros(len(df))
            trend = np.zeros(len(df))
            ep = np.zeros(len(df))
            acc = np.zeros(len(df))
            
            # Initialize first values
            sar[0] = low[0]
            trend[0] = 1  # 1 for uptrend, -1 for downtrend
            ep[0] = high[0]
            acc[0] = af
            
            for i in range(1, len(df)):
                if trend[i-1] == 1:  # Uptrend
                    sar[i] = sar[i-1] + acc[i-1] * (ep[i-1] - sar[i-1])
                    
                    if low[i] <= sar[i]:
                        # Trend reversal
                        trend[i] = -1
                        sar[i] = ep[i-1]
                        ep[i] = low[i]
                        acc[i] = af
                    else:
                        trend[i] = 1
                        if high[i] > ep[i-1]:
                            ep[i] = high[i]
                            acc[i] = min(acc[i-1] + af, max_af)
                        else:
                            ep[i] = ep[i-1]
                            acc[i] = acc[i-1]
                            
                else:  # Downtrend
                    sar[i] = sar[i-1] + acc[i-1] * (ep[i-1] - sar[i-1])
                    
                    if high[i] >= sar[i]:
                        # Trend reversal
                        trend[i] = 1
                        sar[i] = ep[i-1]
                        ep[i] = high[i]
                        acc[i] = af
                    else:
                        trend[i] = -1
                        if low[i] < ep[i-1]:
                            ep[i] = low[i]
                            acc[i] = min(acc[i-1] + af, max_af)
                        else:
                            ep[i] = ep[i-1]
                            acc[i] = acc[i-1]
            
            return {
                f"parabolic_sar_{af}": round(float(sar[-1]), 4),
                f"parabolic_sar_trend_{af}": int(trend[-1])
            }
            
        except Exception as e:
            logger.error(f"Error calculating Parabolic SAR: {e}")
            return {}
    
    def calculate_fibonacci_retracements(self, prices: pd.Series, 
                                       period: int = 20) -> Dict:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            prices: Price series
            period: Lookback period for high/low
            
        Returns:
            Dictionary with Fibonacci levels
        """
        if len(prices) < period:
            return {}
        
        try:
            recent_prices = prices.tail(period)
            high_price = recent_prices.max()
            low_price = recent_prices.min()
            
            diff = high_price - low_price
            
            # Fibonacci retracement levels
            levels = {
                f"fib_100_{period}": round(float(high_price), 4),
                f"fib_78.6_{period}": round(float(high_price - 0.786 * diff), 4),
                f"fib_61.8_{period}": round(float(high_price - 0.618 * diff), 4),
                f"fib_50.0_{period}": round(float(high_price - 0.5 * diff), 4),
                f"fib_38.2_{period}": round(float(high_price - 0.382 * diff), 4),
                f"fib_23.6_{period}": round(float(high_price - 0.236 * diff), 4),
                f"fib_0_{period}": round(float(low_price), 4)
            }
            
            return levels
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci retracements: {e}")
            return {}
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Pivot Points
        
        Args:
            df: DataFrame with high, low, close columns
            
        Returns:
            Dictionary with pivot points and support/resistance levels
        """
        if len(df) < 1:
            return {}
        
        try:
            # Use previous day's data
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            close = df['close'].iloc[-1]
            
            # Pivot point
            pivot = (high + low + close) / 3
            
            # Support and resistance levels
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                "pivot_point": round(float(pivot), 4),
                "resistance_1": round(float(r1), 4),
                "resistance_2": round(float(r2), 4),
                "resistance_3": round(float(r3), 4),
                "support_1": round(float(s1), 4),
                "support_2": round(float(s2), 4),
                "support_3": round(float(s3), 4)
            }
            
        except Exception as e:
            logger.error(f"Error calculating pivot points: {e}")
            return {}
    
    def calculate_support_resistance(self, prices: pd.Series, 
                                   period: int = 20) -> Dict:
        """
        Calculate Support and Resistance levels
        
        Args:
            prices: Price series
            period: Lookback period
            
        Returns:
            Dictionary with support and resistance levels
        """
        if len(prices) < period:
            return {}
        
        try:
            recent_prices = prices.tail(period)
            
            # Simple support/resistance based on recent highs and lows
            resistance = recent_prices.max()
            support = recent_prices.min()
            
            # Calculate multiple support/resistance levels
            price_range = resistance - support
            
            return {
                f"resistance_level_{period}": round(float(resistance), 4),
                f"support_level_{period}": round(float(support), 4),
                f"resistance_2_{period}": round(float(resistance + 0.382 * price_range), 4),
                f"support_2_{period}": round(float(support - 0.382 * price_range), 4),
                f"price_range_{period}": round(float(price_range), 4)
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {}
    
    def calculate_indicators_for_symbol(self, symbol: str, date: str,
                                      lookback_days: int = 252) -> Dict:
        """
        Calculate all analytical indicators for a symbol on a specific date
        
        Args:
            symbol: Stock symbol
            date: Date in YYYY-MM-DD format
            lookback_days: Number of days to look back for calculations
            
        Returns:
            Dictionary with calculated indicators
        """
        logger.info(f"Calculating analytical indicators for {symbol} on {date}")
        
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
            
            # Calculate Bollinger Bands
            for period in self.indicator_configs["bollinger_bands"]:
                bb_indicators = self.calculate_bollinger_bands(close_prices, period)
                indicators.update(bb_indicators)
            
            # Calculate MACD
            for fast, slow, signal in self.indicator_configs["macd"]:
                macd_indicators = self.calculate_macd(close_prices, fast, slow, signal)
                indicators.update(macd_indicators)
            
            # Calculate Stochastic Oscillator
            for k_period, k_smooth, d_smooth in self.indicator_configs["stochastic"]:
                stoch_indicators = self.calculate_stochastic(df, k_period, k_smooth, d_smooth)
                indicators.update(stoch_indicators)
            
            # Calculate CCI
            for period in self.indicator_configs["cci"]:
                cci_indicators = self.calculate_cci(df, period)
                indicators.update(cci_indicators)
            
            # Calculate Ichimoku Cloud
            for conversion, base, span_b in self.indicator_configs["ichimoku"]:
                ichimoku_indicators = self.calculate_ichimoku_cloud(df, conversion, base, span_b)
                indicators.update(ichimoku_indicators)
            
            # Calculate Parabolic SAR
            for af in self.indicator_configs["parabolic_sar"]:
                sar_indicators = self.calculate_parabolic_sar(df, af)
                indicators.update(sar_indicators)
            
            # Calculate Fibonacci Retracements
            for period in self.indicator_configs["fibonacci_retracements"]:
                fib_indicators = self.calculate_fibonacci_retracements(close_prices, period)
                indicators.update(fib_indicators)
            
            # Calculate Pivot Points
            if self.indicator_configs["pivot_points"]:
                pivot_indicators = self.calculate_pivot_points(df)
                indicators.update(pivot_indicators)
            
            # Calculate Support/Resistance
            for period in self.indicator_configs["support_resistance"]:
                sr_indicators = self.calculate_support_resistance(close_prices, period)
                indicators.update(sr_indicators)
            
            logger.info(f"Calculated {len(indicators)-3} analytical indicators for {symbol} on {date}")
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating analytical indicators for {symbol}: {e}")
            return {}
    
    def calculate_and_save_indicators(self, symbol: str = None, date: str = None,
                                    lookback_days: int = 252) -> bool:
        """
        Calculate and save analytical indicators for symbol(s) and date(s)
        
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
            
            logger.info(f"Processing {len(symbols)} symbols for analytical indicators on {date}")
            
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
                    self.db_provider.analytical_indicators_collection_name
                )
                if success:
                    logger.info(f"Successfully saved analytical indicators for "
                              f"{len(all_indicators)} symbols on {date}")
                    return True
                else:
                    logger.error("Failed to save analytical indicators")
                    return False
            else:
                logger.warning(f"No analytical indicators calculated for any symbols on {date}")
                return False
                
        except Exception as e:
            logger.error(f"Error in calculate_and_save_indicators: {e}")
            return False
    
    def get_indicators_summary(self, symbol: str, start_date: str = None,
                             end_date: str = None) -> pd.DataFrame:
        """
        Get a summary of calculated analytical indicators for a symbol
        
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
                self.db_provider.analytical_indicators_collection_name
            )
            
            if df.empty:
                logger.info(f"No analytical indicators found for {symbol}")
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
            logger.error(f"Error getting analytical indicators summary for {symbol}: {e}")
            return pd.DataFrame()
