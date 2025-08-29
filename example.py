#!/usr/bin/env python3
"""
Example script demonstrating the usage of stocks_indicators package

This script shows how to:
1. Initialize the MongoDB provider and calculators
2. Calculate technical indicators for a symbol
3. Calculate analytical indicators for a symbol
4. Save results to MongoDB
5. Retrieve and analyze saved indicators
"""

import logging
from datetime import datetime, timedelta
from stocks_indicators import MongoDBProvider, TechnicalIndicators, AnalyticalIndicators

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

snp500_symbols = ["MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A", "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE", "AEP", "AXP", "AIG", "AMT", "AWK", "AMP", "AME", "AMGN", "APH", "ADI", "AON", "APA", "APO", "AAPL", "AMAT", "APTV", "ACGL", "ADM", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BAX", "BDX", "BRK.B", "BBY", "TECH", "BIIB", "BLK", "BX", "XYZ", "BK", "BA", "BKNG", "BSX", "BMY", "AVGO", "BR", "BRO", "BF.B", "BLDR", "BG", "BXP", "CHRW", "CDNS", "CZR", "CPT", "CPB", "COF", "CAH", "KMX", "CCL", "CARR", "CAT", "CBOE", "CBRE", "CDW", "COR", "CNC", "CNP", "CF", "CRL", "SCHW", "CHTR", "CVX", "CMG", "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CLX", "CME", "CMS", "KO", "CTSH", "COIN", "CL", "CMCSA", "CAG", "COP", "ED", "STZ", "CEG", "COO", "CPRT", "GLW", "CPAY", "CTVA", "CSGP", "COST", "CTRA", "CRWD", "CCI", "CSX", "CMI", "CVS", "DHR", "DRI", "DDOG", "DVA", "DAY", "DECK", "DE", "DELL", "DAL", "DVN", "DXCM", "FANG", "DLR", "DG", "DLTR", "D", "DPZ", "DASH", "DOV", "DOW", "DHI", "DTE", "DUK", "DD", "EMN", "ETN", "EBAY", "ECL", "EIX", "EW", "EA", "ELV", "EMR", "ENPH", "ETR", "EOG", "EPAM", "EQT", "EFX", "EQIX", "EQR", "ERIE", "ESS", "EL", "EG", "EVRG", "ES", "EXC", "EXE", "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FDS", "FICO", "FAST", "FRT", "FDX", "FIS", "FITB", "FSLR", "FE", "FI", "F", "FTNT", "FTV", "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT", "GE", "GEHC", "GEV", "GEN", "GNRC", "GD", "GIS", "GM", "GPC", "GILD", "GPN", "GL", "GDDY", "GS", "HAL", "HIG", "HAS", "HCA", "DOC", "HSIC", "HSY", "HPE", "HLT", "HOLX", "HD", "HON", "HRL", "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX", "IDXX", "ITW", "INCY", "IR", "PODD", "INTC", "IBKR", "ICE", "IFF", "IP", "IPG", "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", "JBHT", "JBL", "JKHY", "J", "JNJ", "JCI", "JPM", "K", "KVUE", "KDP", "KEY", "KEYS", "KMB", "KIM", "KMI", "KKR", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX", "LW", "LVS", "LDOS", "LEN", "LII", "LLY", "LIN", "LYV", "LKQ", "LMT", "L", "LOW", "LULU", "LYB", "MTB", "MPC", "MKTX", "MAR", "MMC", "MLM", "MAS", "MA", "MTCH", "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD", "MGM", "MCHP", "MU", "MSFT", "MAA", "MRNA", "MHK", "MOH", "TAP", "MDLZ", "MPWR", "MNST", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", "NEM", "NWSA", "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS", "NOC", "NCLH", "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "ON", "OKE", "ORCL", "OTIS", "PCAR", "PKG", "PLTR", "PANW", "PSKY", "PH", "PAYX", "PAYC", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM", "PSX", "PNW", "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PEG", "PTC", "PSA", "PHM", "PWR", "QCOM", "DGX", "RL", "RJF", "RTX", "O", "REG", "REGN", "RF", "RSG", "RMD", "RVTY", "ROK", "ROL", "ROP", "ROST", "RCL", "SPGI", "CRM", "SBAC", "SLB", "STX", "SRE", "NOW", "SHW", "SPG", "SWKS", "SJM", "SW", "SNA", "SOLV", "SO", "LUV", "SWK", "SBUX", "STT", "STLD", "STE", "SYK", "SMCI", "SYF", "SNPS", "SYY", "TMUS", "TROW", "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY", "TER", "TSLA", "TXN", "TPL", "TXT", "TMO", "TJX", "TKO", "TTD", "TSCO", "TT", "TDG", "TRV", "TRMB", "TFC", "TYL", "TSN", "USB", "UBER", "UDR", "ULTA", "UNP", "UAL", "UPS", "URI", "UNH", "UHS", "VLO", "VTR", "VLTO", "VRSN", "VRSK", "VZ", "VRTX", "VTRS", "VICI", "V", "VST", "VMC", "WRB", "GWW", "WAB", "WBA", "WMT", "DIS", "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC", "WY", "WSM", "WMB", "WTW", "WDAY", "WYNN", "XEL", "XYL", "YUM", "ZBRA", "ZBH", "ZTS"]
symbols = snp500_symbols

def main():
    """Main example function"""
    
    # Initialize MongoDB provider
    logger.info("Connecting to MongoDB...")
    db_provider = MongoDBProvider(
        host="localhost",
        port=27017,
        database="stocksdb"
    )
    
    # Configure collection names (optional - using defaults)
    db_provider.price_collection_name = "fmp_stock_price_eod"
    db_provider.technical_indicators_collection_name = "fmp_technical_indicators"
    db_provider.analytical_indicators_collection_name = "fmp_analytical_indicators"
    
    # Initialize calculators
    tech_calc = TechnicalIndicators(db_provider)
    analytical_calc = AnalyticalIndicators(db_provider)
    
    # Example date and symbol
    symbol = "AAPL"
    date = "2025-08-26"
    
    try:
        # 1. Get database statistics
        logger.info("Getting database statistics...")
        stats = db_provider.get_database_stats()
        logger.info(f"Database stats: {stats}")
        
        # 2. Check available symbols for the date
        logger.info(f"Getting symbols available for {date}...")
        #symbols = db_provider.get_symbols_for_date(date)
        logger.info(f"Found {len(symbols)} symbols for {date}")
        
        if symbol not in symbols:
            logger.warning(f"{symbol} not found for {date}. Using first available symbol.")
            if symbols:
                symbol = symbols[0]
            else:
                logger.error("No symbols available for processing")
                return
        
        # 3. Get historical price data
        logger.info(f"Getting historical price data for {symbol}...")
        price_df = db_provider.get_historical_prices(symbol, date, lookback_days=100)
        if not price_df.empty:
            logger.info(f"Retrieved {len(price_df)} price records for {symbol}")
            logger.info(f"Date range: {price_df['ds'].min()} to {price_df['ds'].max()}")
        else:
            logger.error(f"No price data found for {symbol}")
            return
        
        # 4. Calculate technical indicators
        logger.info(f"Calculating technical indicators for {symbol} on {date}...")
        tech_indicators = tech_calc.calculate_indicators_for_symbol(symbol, date)
        
        if tech_indicators:
            logger.info(f"Calculated {len(tech_indicators)-3} technical indicators")
            
            # Show some example indicators
            example_indicators = {k: v for k, v in tech_indicators.items() 
                                if k.startswith(('sma_', 'ema_', 'rsi_'))}
            logger.info(f"Example technical indicators: {example_indicators}")
            
            # Save to MongoDB
            success = db_provider.save_indicators(
                [tech_indicators], 
                db_provider.technical_indicators_collection_name
            )
            if success:
                logger.info("Technical indicators saved successfully")
        else:
            logger.warning("No technical indicators calculated")
        
        # 5. Calculate analytical indicators
        logger.info(f"Calculating analytical indicators for {symbol} on {date}...")
        analytical_indicators = analytical_calc.calculate_indicators_for_symbol(symbol, date)
        
        if analytical_indicators:
            logger.info(f"Calculated {len(analytical_indicators)-3} analytical indicators")
            
            # Show some example indicators
            example_indicators = {k: v for k, v in analytical_indicators.items() 
                                if k.startswith(('bb_', 'macd_', 'stoch_'))}
            logger.info(f"Example analytical indicators: {example_indicators}")
            
            # Save to MongoDB
            success = db_provider.save_indicators(
                [analytical_indicators], 
                db_provider.analytical_indicators_collection_name
            )
            if success:
                logger.info("Analytical indicators saved successfully")
        else:
            logger.warning("No analytical indicators calculated")
        
        # 6. Retrieve saved indicators
        logger.info("Retrieving saved technical indicators...")
        saved_tech = db_provider.get_indicators(
            symbol, 
            start_date=date, 
            end_date=date,
            collection_name=db_provider.technical_indicators_collection_name
        )
        
        if not saved_tech.empty:
            logger.info(f"Retrieved {len(saved_tech)} technical indicator records")
            logger.info(f"Columns: {list(saved_tech.columns)}")
        
        logger.info("Retrieving saved analytical indicators...")
        saved_analytical = db_provider.get_indicators(
            symbol,
            start_date=date,
            end_date=date, 
            collection_name=db_provider.analytical_indicators_collection_name
        )
        
        if not saved_analytical.empty:
            logger.info(f"Retrieved {len(saved_analytical)} analytical indicator records")
            logger.info(f"Columns: {list(saved_analytical.columns)}")
        
        # 7. Get indicator summaries
        logger.info("Getting technical indicators summary...")
        tech_summary = tech_calc.get_indicators_summary(symbol)
        if not tech_summary.empty:
            logger.info(f"Technical indicators summary: {tech_summary.to_dict('records')[0]}")
        
        logger.info("Getting analytical indicators summary...")
        analytical_summary = analytical_calc.get_indicators_summary(symbol)
        if not analytical_summary.empty:
            logger.info(f"Analytical indicators summary: {analytical_summary.to_dict('records')[0]}")
        
        # 8. Demonstrate batch processing
        logger.info("Demonstrating batch processing for multiple symbols...")
        if len(symbols) > 1:
            # Process first 3 symbols (or all if less than 3)
            batch_symbols = symbols[:3]
            logger.info(f"Processing batch of symbols: {batch_symbols}")
            
            for sym in batch_symbols:
                try:
                    # Calculate and save technical indicators
                    success = tech_calc.calculate_and_save_indicators(symbol=sym, date=date)
                    if success:
                        logger.info(f"Successfully processed technical indicators for {sym}")
                    
                    # Calculate and save analytical indicators  
                    success = analytical_calc.calculate_and_save_indicators(symbol=sym, date=date)
                    if success:
                        logger.info(f"Successfully processed analytical indicators for {sym}")
                        
                except Exception as e:
                    logger.error(f"Error processing {sym}: {e}")
                    continue
        
        # 9. Create database indexes for performance
        logger.info("Creating database indexes...")
        db_provider.create_indexes()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main example: {e}")
        
    finally:
        # Clean up
        logger.info("Closing database connection...")
        db_provider.close()


if __name__ == "__main__":
    main()
