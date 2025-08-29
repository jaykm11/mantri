"""
Stocks Indicators Package

A self-contained package for calculating technical and analytical indicators
from stock price data stored in MongoDB.

This package provides:
1. MongoDB abstraction layer for data operations
2. Basic technical indicators calculation
3. Advanced analytical indicators calculation (Bollinger Bands, etc.)
"""

from .mongodb_provider import MongoDBProvider
from .technical_indicators import TechnicalIndicators
from .analytical_indicators import AnalyticalIndicators

__version__ = "1.0.0"
__all__ = ["MongoDBProvider", "TechnicalIndicators", "AnalyticalIndicators"]
