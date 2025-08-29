"""
MongoDB Data Provider

A self-contained MongoDB abstraction layer for stock data operations.
Provides methods to read historical price data and save technical indicators.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pandas as pd
from pymongo import MongoClient, UpdateOne, ASCENDING

logger = logging.getLogger(__name__)


class MongoDBProvider:
    """
    MongoDB abstraction layer for stock data operations
    """
    
    def __init__(self, host: str = "localhost", port: int = 27017, 
                 database: str = "stocksdb"):
        """
        Initialize MongoDB connection
        
        Args:
            host: MongoDB host
            port: MongoDB port  
            database: Database name
        """
        self.host = host
        self.port = port
        self.database_name = database
        self.client = MongoClient(host, port)
        self.db = self.client[database]
        
        # Default collection names
        self.price_collection_name = "fmp_stock_price_eod"
        self.technical_indicators_collection_name = "fmp_technical_indicators"
        self.analytical_indicators_collection_name = "fmp_analytical_indicators"
        
        logger.info(f"Connected to MongoDB at {host}:{port}, database: {database}")
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def get_collection(self, collection_name: str):
        """Get MongoDB collection"""
        return self.db[collection_name]
    
    def get_symbols_for_date(self, date: str, 
                           collection_name: str = None) -> List[str]:
        """
        Get all unique symbols that have data for a specific date
        
        Args:
            date: Date string in YYYY-MM-DD format
            collection_name: Name of the collection to query
            
        Returns:
            List of symbols
        """
        if collection_name is None:
            collection_name = self.price_collection_name
            
        try:
            collection = self.get_collection(collection_name)
            symbols = collection.distinct("symbol", {"ds": date})
            logger.debug(f"Found {len(symbols)} symbols for date {date}")
            return symbols
        except Exception as e:
            logger.error(f"Error getting symbols for date {date}: {e}")
            return []
    
    def get_historical_prices(self, symbol: str, end_date: str,
                            lookback_days: int = 252,
                            collection_name: str = None) -> pd.DataFrame:
        """
        Get historical price data for a symbol
        
        Args:
            symbol: Stock symbol
            end_date: End date in YYYY-MM-DD format
            lookback_days: Number of trading days to look back
            collection_name: Name of the price collection
            
        Returns:
            DataFrame with columns: ds, open, high, low, close, adjClose, volume
        """
        if collection_name is None:
            collection_name = self.price_collection_name
            
        try:
            # Calculate start date
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=lookback_days * 2)  # Buffer for weekends
            start_date = start_dt.strftime("%Y-%m-%d")
            
            # Query data
            collection = self.get_collection(collection_name)
            cursor = collection.find({
                "symbol": symbol,
                "ds": {"$gte": start_date, "$lte": end_date}
            }).sort("ds", ASCENDING)
            
            data = list(cursor)
            if not data:
                logger.warning(f"No price data found for {symbol} up to {end_date}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Ensure required columns exist
            required_cols = ['ds', 'open', 'high', 'low', 'close', 'volume']
            if 'adjClose' in df.columns:
                required_cols.append('adjClose')
                
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns for {symbol}: {missing_cols}")
                return pd.DataFrame()
            
            # Convert price columns to numeric
            price_cols = ['open', 'high', 'low', 'close', 'volume']
            if 'adjClose' in df.columns:
                price_cols.append('adjClose')
                
            for col in price_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by date and reset index
            df = df.sort_values('ds').reset_index(drop=True)
            
            logger.debug(f"Retrieved {len(df)} price records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical prices for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_latest_price(self, symbol: str, date: str = None,
                        collection_name: str = None) -> Dict:
        """
        Get the latest price record for a symbol
        
        Args:
            symbol: Stock symbol
            date: Optional date to get price for specific date
            collection_name: Name of the price collection
            
        Returns:
            Dictionary with price data or empty dict if not found
        """
        if collection_name is None:
            collection_name = self.price_collection_name
            
        try:
            collection = self.get_collection(collection_name)
            
            query = {"symbol": symbol}
            if date:
                query["ds"] = date
            
            record = collection.find_one(query, sort=[("ds", -1)])
            return record if record else {}
            
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return {}
    
    def save_indicators(self, indicators: List[Dict], 
                       collection_name: str = None) -> bool:
        """
        Save technical or analytical indicators to MongoDB
        
        Args:
            indicators: List of indicator documents
            collection_name: Target collection name
            
        Returns:
            True if successful, False otherwise
        """
        if collection_name is None:
            collection_name = self.technical_indicators_collection_name
            
        if not indicators:
            logger.warning("No indicators to save")
            return False
        
        try:
            collection = self.get_collection(collection_name)
            
            # Prepare bulk operations for upsert
            operations = []
            for indicator_doc in indicators:
                if not indicator_doc or 'symbol' not in indicator_doc or 'ds' not in indicator_doc:
                    continue
                
                # Create filter for upsert
                filter_query = {
                    "symbol": indicator_doc["symbol"],
                    "ds": indicator_doc["ds"]
                }
                
                # Create update operation
                operations.append(
                    UpdateOne(
                        filter_query,
                        {"$set": indicator_doc},
                        upsert=True
                    )
                )
            
            if operations:
                result = collection.bulk_write(operations)
                logger.info(f"Saved to {collection_name}: "
                          f"{result.upserted_count} inserted, "
                          f"{result.modified_count} updated")
                return True
            else:
                logger.warning("No valid indicators to save")
                return False
                
        except Exception as e:
            logger.error(f"Error saving indicators to {collection_name}: {e}")
            return False
    
    def get_indicators(self, symbol: str, start_date: str = None, 
                      end_date: str = None, collection_name: str = None) -> pd.DataFrame:
        """
        Get saved indicators for a symbol
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            collection_name: Collection to query
            
        Returns:
            DataFrame with indicator data
        """
        if collection_name is None:
            collection_name = self.technical_indicators_collection_name
            
        try:
            collection = self.get_collection(collection_name)
            
            query = {"symbol": symbol}
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = start_date
                if end_date:
                    date_filter["$lte"] = end_date
                query["ds"] = date_filter
            
            cursor = collection.find(query).sort("ds", ASCENDING)
            data = list(cursor)
            
            if not data:
                logger.debug(f"No indicators found for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df = df.sort_values('ds').reset_index(drop=True)
            
            logger.debug(f"Retrieved {len(df)} indicator records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting indicators for {symbol}: {e}")
            return pd.DataFrame()
    
    def delete_indicators(self, symbol: str = None, date: str = None,
                         collection_name: str = None) -> int:
        """
        Delete indicators from collection
        
        Args:
            symbol: Symbol to delete (optional)
            date: Date to delete (optional)
            collection_name: Collection name
            
        Returns:
            Number of deleted documents
        """
        if collection_name is None:
            collection_name = self.technical_indicators_collection_name
            
        try:
            collection = self.get_collection(collection_name)
            
            query = {}
            if symbol:
                query["symbol"] = symbol
            if date:
                query["ds"] = date
            
            if not query:
                logger.warning("No filter provided for deletion - operation cancelled")
                return 0
            
            result = collection.delete_many(query)
            logger.info(f"Deleted {result.deleted_count} documents from {collection_name}")
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting indicators: {e}")
            return 0
    
    def create_indexes(self):
        """Create recommended indexes for performance"""
        try:
            # Indexes for price collection
            price_col = self.get_collection(self.price_collection_name)
            price_col.create_index([("symbol", 1), ("ds", 1)], unique=True)
            price_col.create_index([("ds", 1)])
            
            # Indexes for technical indicators
            tech_col = self.get_collection(self.technical_indicators_collection_name)
            tech_col.create_index([("symbol", 1), ("ds", 1)], unique=True)
            tech_col.create_index([("ds", 1)])
            
            # Indexes for analytical indicators
            anal_col = self.get_collection(self.analytical_indicators_collection_name)
            anal_col.create_index([("symbol", 1), ("ds", 1)], unique=True)
            anal_col.create_index([("ds", 1)])
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            stats = {
                "database": self.database_name,
                "collections": {}
            }
            
            for col_name in [self.price_collection_name, 
                           self.technical_indicators_collection_name,
                           self.analytical_indicators_collection_name]:
                collection = self.get_collection(col_name)
                count = collection.count_documents({})
                stats["collections"][col_name] = {
                    "document_count": count
                }
                
                if count > 0:
                    # Get date range
                    earliest = collection.find_one({}, sort=[("ds", 1)])
                    latest = collection.find_one({}, sort=[("ds", -1)])
                    
                    if earliest and latest:
                        stats["collections"][col_name].update({
                            "earliest_date": earliest.get("ds"),
                            "latest_date": latest.get("ds")
                        })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
