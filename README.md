# Stocks Indicators Package

A self-contained Python package for calculating technical and analytical indicators from stock price data stored in MongoDB.

## Features

- **MongoDB Abstraction Layer**: Easy-to-use interface for stock data operations
- **Technical Indicators**: Basic indicators like SMA, EMA, RSI, MACD, etc.
- **Analytical Indicators**: Advanced indicators like Bollinger Bands, Ichimoku Cloud, Fibonacci retracements, etc.
- **Self-contained**: No external dependencies on local modules

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Setup

```python
from stocks_indicators import MongoDBProvider, TechnicalIndicators, AnalyticalIndicators

# Initialize MongoDB provider
db_provider = MongoDBProvider(
    host="localhost",
    port=27017,
    database="stocksdb"
)

# Set collection names (optional - defaults provided)
db_provider.price_collection_name = "stock_prices"
db_provider.technical_indicators_collection_name = "technical_indicators"
db_provider.analytical_indicators_collection_name = "analytical_indicators"

# Initialize calculators
tech_calc = TechnicalIndicators(db_provider)
analytical_calc = AnalyticalIndicators(db_provider)
```

### Calculate Technical Indicators

```python
# Calculate for a specific symbol and date
indicators = tech_calc.calculate_indicators_for_symbol("AAPL", "2025-08-26")
print(indicators)

# Calculate and save to MongoDB
success = tech_calc.calculate_and_save_indicators(
    symbol="AAPL",
    date="2025-08-26",
    lookback_days=252
)

# Process all symbols for a date
success = tech_calc.calculate_and_save_indicators(date="2025-08-26")
```

### Calculate Analytical Indicators

```python
# Calculate Bollinger Bands, MACD, etc. for a symbol
indicators = analytical_calc.calculate_indicators_for_symbol("AAPL", "2025-08-26")
print(indicators)

# Calculate and save to MongoDB
success = analytical_calc.calculate_and_save_indicators(
    symbol="AAPL",
    date="2025-08-26"
)
```

### MongoDB Operations

```python
# Get historical price data
df = db_provider.get_historical_prices("AAPL", "2025-08-26", lookback_days=100)

# Get symbols for a specific date
#symbols = db_provider.get_symbols_for_date("2025-08-26")
symbols = db_provider.symbols

# Get saved indicators
indicators_df = db_provider.get_indicators(
    "AAPL", 
    start_date="2025-08-01", 
    end_date="2025-08-26",
    collection_name="technical_indicators"
)

# Create database indexes for performance
db_provider.create_indexes()

# Get database statistics
stats = db_provider.get_database_stats()
print(stats)
```

## Data Format

### Historical Price Data Format
The package expects price data in the following format:

```json
{
    "symbol": "AMZN",
    "date": "2025-08-22",
    "open": 222.79,
    "low": 220.82,
    "high": 229.14,
    "close": 228.84,
    "adjClose": 228.84,
    "volume": 37315341.0,
    "apipath": "/api/v4/batch-request-end-of-day-prices?date=2025-08-22",
    "ds": "2025-08-22"
}
```

## Technical Indicators Supported

### Moving Averages
- Simple Moving Average (SMA): 10, 20, 50, 200 periods
- Exponential Moving Average (EMA): 10, 20, 50, 200 periods
- Weighted Moving Average (WMA): 10, 20 periods
- Double Exponential Moving Average (DEMA): 10, 20 periods
- Triple Exponential Moving Average (TEMA): 10, 20 periods

### Momentum Indicators
- Relative Strength Index (RSI): 14, 20 periods
- Average Directional Index (ADX): 14, 20 periods
- Williams %R: 14, 20 periods
- Momentum: 10, 20 periods
- Rate of Change (ROC): 10, 20 periods

### Volatility Indicators
- Standard Deviation: 10, 20 periods
- Average True Range (ATR): 14, 20 periods

## Analytical Indicators Supported

### Trend Analysis
- Bollinger Bands (upper, middle, lower, bandwidth, %B)
- MACD (line, signal, histogram)
- Ichimoku Cloud (conversion, base, span A/B, lagging)
- Parabolic SAR

### Oscillators
- Stochastic Oscillator (%K, %D)
- Commodity Channel Index (CCI)

### Support/Resistance
- Fibonacci Retracements (0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%)
- Pivot Points (pivot, support 1-3, resistance 1-3)
- Support/Resistance levels

## Configuration

You can customize the indicators calculated by modifying the `indicator_configs` in the calculator classes:

```python
# Customize technical indicators
tech_calc.indicator_configs["sma"] = [5, 10, 21, 50]  # Custom SMA periods
tech_calc.indicator_configs["rsi"] = [14]             # Only RSI-14

# Customize analytical indicators
analytical_calc.indicator_configs["bollinger_bands"] = [20, 30]  # Multiple BB periods
analytical_calc.indicator_configs["macd"] = [(12, 26, 9), (5, 35, 5)]  # Multiple MACD configs
```

## Error Handling

The package includes comprehensive error handling and logging. Enable logging to see detailed information:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('stocks_indicators')
```

## Performance Tips

1. **Create Database Indexes**: Use `db_provider.create_indexes()` for better query performance
2. **Batch Processing**: Process multiple symbols at once rather than one by one
3. **Lookback Period**: Adjust `lookback_days` based on your indicator requirements
4. **Connection Management**: Close database connections when done: `db_provider.close()`

## License

This package is self-contained and designed for internal use with stock analysis systems.
