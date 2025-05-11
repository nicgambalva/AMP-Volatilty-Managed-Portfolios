# This file part of the Master Project for the Msc in Financial Engineering at EDHEC Buusiness School
# Property of Wiktor Kotwicki, Moana Valdenaire, and Nicolas Gamboa Alvarez
# Copyright (c) 2024-2025 Wiktor Kotwicki, Moana Valdenaire, and Nicolas Gamboa Alvarez
# EDHEC Business School, 2024-2025

# This file builds classes necessary to build and test portfolio construction techniques
# applied to futures, particularly futures on equity indexes and commodities.
# The classes are:
# - contract: A class that represents a particular futures contract
# - futures: A class that represents a collections of futures contracts with the same underlying asset
# - strategy: A class that represents a time series of the implementation of a portfolio construction technique

# It also contains common functions used throughout the project to process, analyze, and visualize data.
# The classes build upon each other, where a strategy contains a collection of futures, each future contains a collection of contracts.
# These classes have been optimized to use data from Bloomberg as organized through an Excel add-in, but they can be used with any data source.


# ----------- IMPORTS -----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from numba import njit
from joblib import Parallel, delayed

# ----------- DICTIONARIES -----------

# Dictionary for European Index Futures being considered
# The keys are the names of the indices, and the values are the corresponding Bloomberg tickers for the first generic future contract
FUTURES_EURO_INDICES = {
    'DAX30': 'GX1 Index', # Germany: DAX 30
    'CAC40': 'CF1 Index', # France: CAC 40
    'FTSE Athens 20': 'AJ1 Index', # Greece: FTSE/ATHEX 20
    'FTSE MIB': 'ST1 Index', # Italy: FTSE MIB
    'AEX': 'EO1 Index', # Netherlands: AEX
    'PSI 20': 'PP1 Index', # Portugal: PSI 20
    'IBEX35': 'IB1 Index', # Spain: IBEX 35
    'BEL20': 'BE1 Index' # Belgium: BEL 20
}

# Dictionary for all metal futures being considered
FUTURES_METALS = {
    'Gold': 'GC1 Comdty', # Gold - CMX-COMEX division of NYMEX
    'Silver': 'SI1 Comdty', # Silver - CMX-COMEX division of NYMEX
    'Platinum': 'PL1 Comdty', # Platinum - NYM-NYMEX Exchange
    'Palladium': 'PA1 Comdty', # Palladium - NYM-NYMEX Exchange
    'Nickel': 'LN1 Comdty', # Nickel - LME-LME Benchmark Monitor
    'Zinc': 'LX1 Comdty', # Zinc - LME-LME Benchmark Monitor
    'Tin': 'LT1 Comdty', # Tin - LME-LME Benchmark Monitor
    'Copper': 'LP1 Comdty', # Copper - LME-LME Benchmark Monitor
    'Aluminium': 'LA1 Comdty' # Aluminium - LME-LME Benchmark Monitor
}

# Dictionary with the supported metals futures
SUPPORTED_METALS = {
    'Gold': 'GC1 Comdty', # Gold - CMX-COMEX division of NYMEX
    'Silver': 'SI1 Comdty' # Silver - CMX-COMEX division of NYMEX
}

# Dictionary of possible price types
SUPPORTED_PRICE_TYPES = {
    'PX_BID': 'Bid Price',
    'PX_ASK': 'Ask Price',
    'PX_SETTLE': 'Settlement Price',
    'PX_LAST': 'Last Price',
}

# Dictionary of possible contract types
SUPPORTED_CONTRACT_TYPES = {
    'INDEX': 'Index Future',
    'METAL': 'Metal Future',
}

# Dictionary of possible currencies
SUPPORTED_CURRENCIES = {
    'EUR': 'Euro',
    'USD': 'US Dollar'
}

# Dictionary of possible strategies - For the construction of weights
SUPPORTED_STRATEGIES = {
    'EQUAL_WEIGHTED': 'Equal Weighted',
    'INVERSE_VARIANCE': 'Inverse Variance'
}

# Dictionary of supported rebalancing frequencies
SUPPORTED_REBAL_FREQS = {
    '1M': '1 Month',
    '3M': '3 Months',
    '6M': '6 Months',
    '12M': '12 Months'
}

# Dictionary of supported estimation windows for volatility
SUPPORTED_ESTIMATION_WINDOWS = {
    '1M': 21, # 1 Month
    '3M': 63, # 3 Months
    '6M': 126, # 6 Months
    '12M': 252 # 12 Months
}

# ----------- FUNCTIONS -----------

# Function that does preprocessing of the data
# Particularly optimized for the data as it is being structure for this project using Bloomberg
# Preprocessing the data:
# 1. DATE column is converted to datetime format %m/%d/%Y
# 2. The first row is removed as it is usually saved as an error #NAME
# 3. The DATE column is set as the index
# 4. BBG uses '#N/A N/A' to indicate missing values, which is replaced with np.nan
def preprocess_data(df):
    """Preprocesses the data by converting the DATE column to datetime format, 
    removing the first row, setting the DATE column as index, and replacing 
    '#N/A N/A' with np.nan."""
    df = df.copy()
    if 'DATE' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'DATE' column.")
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.dropna(subset=['DATE'])
    df = df.iloc[1:].copy()
    df.set_index('DATE', inplace=True)
    df.replace('#N/A N/A', np.nan, inplace=True)
    return df

# Function that identifies the future based on the name
# It does so based on the first two letters of the name
# It is mostly used to take a contract name and identify the future it belongs to
# It takes the name, and also the class of the future (As of now, only 'INDEX' and 'METAL')
def identify_future(name, future_class = None):
    """Identifies the future based on the name and the class of the future."""
    if future_class not in ['INDEX', 'METAL', None]:
        raise ValueError("Future class must be 'INDEX', 'METAL', or None.")
    identifier = name[:2]
    if future_class == 'INDEX':
        for future in FUTURES_EURO_INDICES.values():
            if future[:2] == identifier:
                return future
    elif future_class == 'METAL':
        for future in FUTURES_METALS.values():
            if future[:2] == identifier:
                return future
    else:
        for future in FUTURES_EURO_INDICES.values():
            if future[:2] == identifier:
                return future
        for future in FUTURES_METALS.values():
            if future[:2] == identifier:
                return future
    raise ValueError(f"Future {name} not found in any data.")

# A function that gets a specific contravct data type from data defined in the environment
# It takes the name of the contract and the data type as arguments
# It returns the data as a pandas DataFrame with no NaN values
def get_contract_data(contract_ticker, price_type):
    """Gets the contract data for a specific contract and price type."""
    # Assuming dataframes_PX_BID, dataframes_PX_ASK, and dataframes_PX_SETTLE are defined globally
    global dataframes_PX_BID, dataframes_PX_ASK, dataframes_PX_SETTLE
    possible_price_types = ['PX_BID', 'PX_ASK', 'PX_SETTLE']
    if price_type not in possible_price_types:
        raise ValueError(f"Invalid price type. Choose from {possible_price_types}.")
    
    found = False
    if price_type == 'PX_BID':
        for df in dataframes_PX_BID:
            if contract_ticker in df.columns:
                contract_data = df[contract_ticker]
                found = True
                break
    elif price_type == 'PX_ASK':
        for df in dataframes_PX_ASK:
            if contract_ticker in df.columns:
                contract_data = df[contract_ticker]
                found = True
                break
    elif price_type == 'PX_SETTLE':
        for df in dataframes_PX_SETTLE:
            if contract_ticker in df.columns:
                contract_data = df[contract_ticker]
                found = True
                break
    if not found:
        raise ValueError(f"Contract {contract_ticker} not found in {price_type} data.")
    contract_data = contract_data.dropna()
    contract_data = contract_data.sort_index()
    return contract_data

def plot_weights(weights_df, title='Weights overtime', figsize=(15, 10), dpi=300):
    """
    Plots the weights for a given strategy as a stacked bar chart.

    Parameters:
    - weights_df (pd.DataFrame): DataFrame containing the weights with dates as the index.
    - title (str): Title of the plot.
    - figsize (tuple): Size of the figure.
    - dpi (int): Dots per inch for rendering quality.
    """
    # Create a continuous date range
    full_date_range = pd.date_range(start=weights_df.index.min(), end=weights_df.index.max(), freq='D')

    # Reindex the DataFrame to include all dates
    weights_df_cleaned = weights_df.reindex(full_date_range)

    # Fill missing values (e.g., forward-fill)
    weights_df_cleaned.fillna(method='ffill', inplace=True)

    # Plotting
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.suptitle(title, fontsize=20)

    # Stacked bar plot
    for idx, contract in enumerate(weights_df_cleaned.columns):
        ax.bar(weights_df_cleaned.index, weights_df_cleaned[contract], label=contract, 
               bottom=weights_df_cleaned.iloc[:, :idx].sum(axis=1) if idx > 0 else None)

    # Adding a horizontal line at 0 and at 1
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(1, color='black', linewidth=0.8)

    # Formatting
    ax.set_title('Weights Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weights')
    ax.legend(title='Futures', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim([weights_df_cleaned.index[0], weights_df_cleaned.index[-1]])

    plt.tight_layout()
    plt.show()

def downside_volatility(returns: pd.Series) -> float:
    """
    Calculate downside volatility (semi-deviation)
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("Returns must be a pandas Series.")
    
    # Replace positive returns with 0
    modified_returns = returns.copy()
    modified_returns[modified_returns > 0] = 0
    
    return np.sqrt(np.mean(modified_returns ** 2))

@njit
def downside_volatility_numba(returns: np.ndarray) -> float:
    """
    Calculate downside volatility (semi-deviation) using Numba for performance
    """
    if not isinstance(returns, np.ndarray):
        raise ValueError("Returns must be a numpy array.")
    
    # Replace positive returns with 0
    modified_returns = np.where(returns > 0, 0, returns)
    return np.sqrt(np.mean(modified_returns ** 2))


def garch(window_returns, foreast_horizon=1 ,p=1 ,q=1):
    """
    Apply GARCH(p,q) model to a rolling window of returns and forecast volatility.
    Designed to be used with pd.DataFrame()
    """
    # Skip if window has NaN values (incomplete window)
    if window_returns.isna().any():
        return np.nan
    
    try:
        # Fit GARCH model
        model = arch_model(window_returns, vol='Garch', p=p, q=q)
        fitted_model = model.fit(disp='off')
        
        # Forecast specified horizon ahead
        forecast = fitted_model.forecast(horizon=foreast_horizon)
        
        # Return the full forecast or just the first step
        if foreast_horizon == 1:
            return forecast.variance.values[-1, 0]
        else:
            return forecast.variance.values[-1, :]
    
    except Exception as e:
        print(f"Error in GARCH fitting: {e}")
        return np.nan

def rolling_garch_parallel(returns: pd.Series, window: int, n_jobs: int = -1, **garch_kwargs):
    """
    Compute rolling GARCH volatility in parallel using joblib.
    returns: pd.Series of returns
    window: rolling window size
    n_jobs: number of parallel jobs (-1 = all CPUs)
    garch_kwargs: arguments for garch()
    Returns: pd.Series of GARCH volatility estimates
    """
    # Prepare rolling windows
    windows = [returns.iloc[i-window:i] for i in range(window, len(returns)+1)]
    indices = returns.index[window-1:]

    # Run GARCH in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(garch)(w, **garch_kwargs) for w in windows
    )

    # Build result series
    garch_series = pd.Series(results, index=indices)
    # Optionally, forward-fill to match original index length
    garch_series = garch_series.reindex(returns.index)
    return garch_series

# ----------- CLASSES -----------

class Contract:
    """
    A class that represents a particular futures contract.
    It can handle both index and metal futures (with the potential to add more in the future). 
    """
    def __init__(self, name: str, type: str, currency: str, calendar: dict, px_settle: pd.Series = None, px_bid: pd.Series = None, px_ask: pd.Series = None, underlying_data: pd.Series = None):
        self.name = name
        self.type = type
        self.currency = currency
        self.PX_SETTLE = px_settle
        self.PX_BID = px_bid
        self.PX_ASK = px_ask
        self.underlying_data = underlying_data
        self.calendar = calendar
        
        try:
            pass
            self.underlying = identify_future(name, type)
        except ValueError:
            raise ValueError(f"Contract {name} not found in any data.")
        
        # Explicit type checking
        if not isinstance(self.name, str):
            raise TypeError("Contract name must be a string.")
        if not isinstance(self.calendar, dict):
            raise TypeError("Calendar must be a dictionary.")
        if not isinstance(self.type, str):
            raise TypeError("Contract type must be a string.")
        if not isinstance(self.currency, str):
            raise TypeError("Contract currency must be a string.")
        if self.type not in SUPPORTED_CONTRACT_TYPES:
            raise ValueError(f"Contract type {self.type} not supported. Supported types are: {list(SUPPORTED_CONTRACT_TYPES.keys())}.")
        if self.currency not in SUPPORTED_CURRENCIES:
            raise ValueError(f"Contract currency {self.currency} not supported. Supported currencies are: {list(SUPPORTED_CURRENCIES.keys())}.")
        if self.PX_SETTLE is not None and not isinstance(self.PX_SETTLE, pd.Series):
            raise TypeError("PX_SETTLE must be a pandas Series.")
        if self.PX_BID is not None and not isinstance(self.PX_BID, pd.Series):
            raise TypeError("PX_BID must be a pandas Series.")
        if self.PX_ASK is not None and not isinstance(self.PX_ASK, pd.Series):
            raise TypeError("PX_ASK must be a pandas Series.")
        if self.underlying_data is not None and not isinstance(self.underlying_data, pd.Series):
            raise TypeError("Underlying data must be a pandas Series.")
        
        # Trying to get the contract data from the calendar
        try:
            self.start_date = self.calendar[self.underlying]['Start Date'][self.name]           
            self.last_trade_date = self.calendar[self.underlying]['Last Trade'][self.name]
            self.first_notice_date = self.calendar[self.underlying]['First Notice'][self.name]
            self.last_delivery_date = self.calendar[self.underlying]['Last Delivery'][self.name]
        except KeyError:
            raise ValueError(f"Contract {name} not found in calendar data, or the calendar data is not structured correctly or incomplete.")
        
        # If no data is provided, try to get it from the dataframes
        if self.PX_SETTLE is None:
            try:
                self.PX_SETTLE = get_contract_data(name, 'PX_SETTLE')
            except ValueError:
                self.PX_SETTLE = None
        if self.PX_BID is None:
            try:
                self.PX_BID = get_contract_data(name, 'PX_BID')
            except ValueError:
                self.PX_BID = None
        if self.PX_ASK is None:
            try:
                self.PX_ASK = get_contract_data(name, 'PX_ASK')
            except ValueError:
                self.PX_ASK = None
        if self.PX_SETTLE is None and self.PX_BID is None and self.PX_ASK is None:
            raise ValueError(f"Contract {name} not found in any data, this contract cannot be created.")
        
        # Checking that the underlying data is a series with a datetime index and one column
        if self.underlying_data is not None:

            if isinstance(self.underlying_data, pd.Series) and self.underlying_data.index.dtype == 'datetime64[ns]':
                self.underlying_data = self.underlying_data
            else:
                raise ValueError("Underlying data must be a Series with a datetime index.")
        
        if px_settle is None or px_settle.empty:
            raise ValueError(f"PX_SETTLE data is missing for contract {contract}")
    
    ###########################################################################
    
    def __repr__(self):
        return f"Contract: {self.name}, Underlying: {self.underlying}, Start Date: {self.start_date}, Last Trade Date: {self.last_trade_date}, First Notice Date: {self.first_notice_date}, Last Delivery Date: {self.last_delivery_date}"
    def __str__(self):
        return f"Contract({self.name}, for {self.underlying}, from {self.start_date} to {self.last_trade_date})"
    def __hash__(self):
        return hash(self.name)


class Currency:
    """A class that represents a currency. This is made to organize the data for foreign exchange rates."""
    def __init__(self, name: str, base_currency: str, quote_currency: str, px_bid: pd.Series, px_ask: pd.Series, px_last: pd.Series):
        self.name = name
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.px_bid = px_bid
        self.px_ask = px_ask
        self.px_last = px_last
        
        # Explicit type checking
        if not isinstance(self.name, str):
            raise TypeError("Currency name must be a string.")
        if not isinstance(self.base_currency, str):
            raise TypeError("Base currency must be a string.")
        if not isinstance(self.quote_currency, str):
            raise TypeError("Quote currency must be a string.")
        if self.px_bid is not None and not isinstance(self.px_bid, pd.Series):
            raise TypeError("PX_BID must be a pandas Series.")
        if self.px_ask is not None and not isinstance(self.px_ask, pd.Series):
            raise TypeError("PX_ASK must be a pandas Series.")
        if self.px_last is not None and not isinstance(self.px_last, pd.Series):
            raise TypeError("PX_LAST must be a pandas Series.")
        
        # Dropping all the prices that do not have an associated date, or that value is NaT
        if self.px_bid is not None:
            self.px_bid = self.px_bid.dropna()
            self.px_bid = self.px_bid[~self.px_bid.index.isna()]
            self.px_bid = self.px_bid[~self.px_bid.index.isin([pd.NaT])]
            self.px_bid = self.px_bid[~self.px_bid.index.duplicated(keep='first')]
        if self.px_ask is not None:
            self.px_ask = self.px_ask.dropna()
            self.px_ask = self.px_ask[~self.px_ask.index.isna()]
            self.px_ask = self.px_ask[~self.px_ask.index.isin([pd.NaT])]
            self.px_ask = self.px_ask[~self.px_ask.index.duplicated(keep='first')]
        if self.px_last is not None:
            self.px_last = self.px_last.dropna()
            self.px_last = self.px_last[~self.px_last.index.isna()]
            self.px_last = self.px_last[~self.px_last.index.isin([pd.NaT])]
            self.px_last = self.px_last[~self.px_last.index.duplicated(keep='first')]
        
        self.start_date = None
        self.end_date = None
        
        # We try to get the first common date from the data (px_bid, px_ask, px_last)
        # And the last common date from the data (px_bid, px_ask, px_last)
        # Then we filter the data to only include the dates between the first and last common date
        try:
            first_bid = self.px_bid.index[0]
            first_ask = self.px_ask.index[0]
            first_last = self.px_last.index[0]
            
            last_bid = self.px_bid.index[-1]
            last_ask = self.px_ask.index[-1]
            last_last = self.px_last.index[-1]
            
            self.start_date = max(first_bid, first_ask, first_last)
            self.end_date = min(last_bid, last_ask, last_last)
            
            self.px_bid = self.px_bid[(self.px_bid.index >= self.start_date) & (self.px_bid.index <= self.end_date)]
            self.px_ask = self.px_ask[(self.px_ask.index >= self.start_date) & (self.px_ask.index <= self.end_date)]
            self.px_last = self.px_last[(self.px_last.index >= self.start_date) & (self.px_last.index <= self.end_date)]
        except Exception as e:
            raise ValueError(f"Error getting the first and last common date from the data: {e}")
        
    ############################################################################
    
    def __repr__(self):
        return f"FX: {self.name}, Base Currency: {self.base_currency}, Quote Currency: {self.quote_currency}, Start Date: {self.start_date}, Last Trade Date: {self.end_date}"
    def __str__(self):
        return f"Currency({self.name}, {self.base_currency}/{self.quote_currency}, from {self.start_date} to {self.end_date})"
    def __hash__(self):
        return hash(self.name)

class Future:
    """A class that represents a collection of futures contracts with the same underlying asset.
    It can handle both index and metal futures (with the potential to add more in the future)."""
    def __init__(self, name: str, type: str, currency: str, calendar: dict, underlying_data: pd.Series = None, currency_object: Currency = None):
        self.name = name
        self.type = type
        self.currency = currency
        self.calendar = calendar
        try:
            self.underlying = identify_future(name, type)
        except ValueError:
            raise ValueError(f"Future {name} not found in any data.")
        self.contracts = []
        self.contracts_dict = {}
        self.underlying_data = underlying_data
        self.roll_settle_theoretical = None
        self.roll_settle_theoretical_returns = None
        self.roll_settle_theoretical_log_returns = None
        
        self.realized_vol_roll_1MROLL = None
        self.realized_vol_roll_3MROLL = None
        self.realized_vol_roll_6MROLL = None
        self.realized_vol_roll_12MROLL = None
        
        self.realized_vol_roll_log_1MROLL = None
        self.realized_vol_roll_log_3MROLL = None
        self.realized_vol_roll_log_6MROLL = None
        self.realized_vol_roll_log_12MROLL = None
        
        self.realized_vol_undr_1MROLL = None
        self.realized_vol_undr_3MROLL = None
        self.realized_vol_undr_6MROLL = None
        self.realized_vol_undr_12MROLL = None
        
        self.realized_vol_undr_log_1MROLL = None
        self.realized_vol_undr_log_3MROLL = None
        self.realized_vol_undr_log_6MROLL = None
        self.realized_vol_undr_log_12MROLL = None

        self.realized_downside_vol_undr_1MROLL = None
        self.realized_downside_vol_undr_3MROLL = None
        self.realized_downside_vol_undr_6MROLL = None
        self.realized_downside_vol_undr_12MROLL = None

        self.garch_volatility_1MROLL = None
        self.garch_volatility_3MROLL = None
        self.garch_volatility_6MROLL = None
        self.garch_volatility_12MROLL = None
        
        self.currency_object = currency_object
        if self.currency_object is not None:
            if not isinstance(currency_object, Currency):
                raise ValueError("Currency object must be an instance of Currency.")

            self.base_currency = currency_object.base_currency
            self.quote_currency = currency_object.quote_currency
            self.currency_dates = currency_object.px_last.index
        
        # Explicit type checking
        if not isinstance(self.name, str):
            raise TypeError("Future name must be a string.")
        if not isinstance(self.calendar, dict):
            raise TypeError("Calendar must be a dictionary.")
        if not isinstance(self.type, str):
            raise TypeError("Future type must be a string.")
        if not isinstance(self.currency, str):
            raise TypeError("Future currency must be a string.")
        if self.type not in SUPPORTED_CONTRACT_TYPES:
            raise ValueError(f"Future type {self.type} not supported. Supported types are: {list(SUPPORTED_CONTRACT_TYPES.keys())}.")
        if self.currency not in SUPPORTED_CURRENCIES:
            raise ValueError(f"Future currency {self.currency} not supported. Supported currencies are: {list(SUPPORTED_CURRENCIES.keys())}.")
        if self.underlying_data is not None and not isinstance(self.underlying_data, pd.Series):
            raise TypeError("Underlying data must be a pandas Series.")
        
        # Checking that the calendar has the necessary columns
        #required_columns = ['Start Date', 'Last Trade', 'First Notice', 'Last Delivery']
        #for col in required_columns:
        #    if col not in self.calendar.columns:
        #        raise ValueError(f"Calendar must contain the column {col}.")
        
        # Checking that the name of the future is part of either the index or the metal futures
        # Otherwise raise an error saying that the future might not yet be implemented
        if self.type == 'INDEX':
            if self.name not in FUTURES_EURO_INDICES.values():
                raise ValueError(f"Future {self.name} not found in any data, this future cannot be created.")
        elif self.type == 'METAL':
            if self.name not in FUTURES_METALS.values():
                raise ValueError(f"Future {self.name} not found in any data, this future cannot be created.")
            
        # If type 'METAL' and underlyind data has been provided, raising an error
        if self.type == 'METAL' and self.underlying_data is not None:
            raise ValueError("Underlying data cannot be provided for metal futures, since best underlying data is the future itself.")
    
    ###########################################################################
        
    def __repr__(self):
        return f"Future: {self.name}, Underlying: {self.underlying}, Number of Contracts: {len(self.contracts)}"
    def __str__(self):
        return f"Future({self.name}, for {self.underlying}, with {len(self.contracts)} contracts)"
    def __hash__(self):
        return hash(self.name)
    
    ###########################################################################
    
    def add_contract(self, contract_obj):
        """Adds a contract to the future."""
        if isinstance(contract_obj, Contract):
            if contract_obj.underlying != self.underlying:
                raise ValueError(f"Contract {contract_obj.name} does not belong to this future {self.name}.")
            if contract_obj.name in self.contracts_dict:
                raise ValueError(f"Contract {contract_obj.name} already exists in this future {self.name}.")
            if contract_obj.type != self.type:
                raise ValueError(f"Contract {contract_obj.name} type {contract_obj.type} does not match future {self.name} type {self.type}.")
            if contract_obj.currency != self.currency:
                raise ValueError(f"Contract {contract_obj.name} currency {contract_obj.currency} does not match future {self.name} currency {self.currency}.")
            self.contracts.append(contract_obj)
            self.contracts_dict[contract_obj.name] = contract_obj
            
        else:
            raise ValueError("Contract must be an instance of contract.")
        
    def remove_contract(self, contract_name):
        """Removes a contract from the future."""
        if contract_name in self.contracts_dict:
            contract = self.contracts_dict[contract_name]
            self.contracts.remove(contract)
            del self.contracts_dict[contract_name]
        else:
            raise ValueError(f"Contract {contract_name} does not exist in this future {self.name}.")
        
    def get_contract(self, contract_name):
        """Gets a contract from the future."""
        if contract_name in self.contracts_dict:
            return self.contracts_dict[contract_name]
        else:
            raise ValueError(f"Contract {contract_name} does not exist in this future {self.name}.")
        
    def get_relevant_contract(self, date: pd.Timestamp, date_delta: int = 0, maturity_delta: int = 0):
        """
        Get the next maturity contract for a given date.
        The date is in datetime format.
        The date_delta is the number of days to add to the date. This is useful if we want to roll before the maturity date.
        So the effective contract we can invest in is the one that is the closest to the date + date_delta.
        The maturity_delta can be used to get not the first next contract, but the second, third, etc.
        """
        try:
            date = pd.to_datetime(date)
        except ValueError:
            raise ValueError("Date must be in datetime format.")
        if date_delta < 0:
            raise ValueError("Date delta must be positive.")
        if date_delta > 0:
            date = date + pd.Timedelta(days=date_delta)
        if maturity_delta < 0:
            raise ValueError("Maturity delta must be positive.")
        
        if len(self.contracts) == 0:
            raise ValueError(f"No contracts available for future {self.name}.")
        if len(self.contracts) == 1:
            return self.contracts[0]
        
        # Sort the contracts by start date
        contracts_sorted = sorted(self.contracts, key=lambda x: x.last_trade_date)
        # Find the next contract
        for contract in contracts_sorted:
            contracts_found = 0
            if contract.last_trade_date > date:
                contracts_found += 1
                if contracts_found == (maturity_delta + 1):
                    return contract
                
        # If no contract is found, we raise an error
        raise ValueError(f"No contract found for future {self.name} for date {date}.")
    
    def get_data_date(self, date: pd.Timestamp, price_type: str):
        """
        Get the data for a given date.
        The date is in datetime format.
        The price_type is the type of price (PX_BID, PX_ASK, PX_SETTLE).
        Just checking, if you're reading this, you can claim a bubble tea from me. No harsh feelings if you don't
        I probably would not read this comment either ...
        """
        try:
            date = pd.to_datetime(date)
        except ValueError:
            raise ValueError("Date must be in datetime format.")
        
        if price_type not in ['PX_BID', 'PX_ASK', 'PX_SETTLE']:
            raise ValueError("Price type must be one of ['PX_BID', 'PX_ASK', 'PX_SETTLE'].")
        
        data = {}
        for contract in self.contracts:
            if price_type == 'PX_BID':
                data[contract.name] = contract.PX_BID.loc[date]
            elif price_type == 'PX_ASK':
                data[contract.name] = contract.PX_ASK.loc[date]
            elif price_type == 'PX_SETTLE':
                data[contract.name] = contract.PX_SETTLE.loc[date]
        
        return data
    
    def get_relevant_maturity_date(self, date: pd.Timestamp, date_delta: int = 0, maturity_delta: int = 0):
        """
        Get the next maturity date for a given date.
        The date is in datetime format.
        The date_delta is the number of days to add to the date. This is useful if we want to roll before the maturity date.
        So the effective contract we can invest in is the one that is the closest to the date + date_delta.
        The maturity_delta can be used to get not the first next contract, but the second, third, etc.
        """
        try:
            date = pd.to_datetime(date)
        except ValueError:
            raise ValueError("Date must be in datetime format.")
        if date_delta < 0:
            raise ValueError("Date delta must be positive.")
        if date_delta > 0:
            date = date + pd.Timedelta(days=date_delta)
        if maturity_delta < 0:
            raise ValueError("Maturity delta must be positive.")
        
        if len(self.contracts) == 0:
            raise ValueError(f"No contracts available for future {self.name}.")
        if len(self.contracts) == 1:
            return self.contracts[0].last_trade_date
        
        # Sort the contracts by start date
        contracts_sorted = sorted(self.contracts, key=lambda x: x.last_trade_date)
        # Find the next contract
        for contract in contracts_sorted:
            contracts_found = 0
            if contract.last_trade_date > date:
                contracts_found += 1
                if contracts_found == (maturity_delta + 1):
                    return contract.last_trade_date
                
        # If no contract is found, return None
        return None
    
    def is_relevant_maturity_date(self, date: pd.Timestamp, date_delta: int = 0, maturity_delta: int = 0):
        """
        Check if a given date is a maturity date for any of the contracts.
        The date is in datetime format.
        """
        try:
            date = pd.to_datetime(date)
        except ValueError:
            raise ValueError("Date must be in datetime format.")
        if date_delta < 0:
            raise ValueError("Date delta must be positive.")
        if date_delta > 0:
            date = date + pd.Timedelta(days=date_delta)
        
        for contract in self.contracts:
            if contract.last_trade_date == date:
                return True
        return False
    
    def get_first_data_date(self):
        """
        Get the first date for the data of the future settlement.
        The date is in datetime format.
        """
        if len(self.contracts) == 0:
            raise ValueError(f"No contracts available for future {self.name}.")
        contracts_sorted = sorted(self.contracts, key=lambda x: x.start_date)
        first_contract = contracts_sorted[0]
        try:
            if self.currency_object is not None:
                if first_contract.start_date < self.currency_dates[0]:
                    first_contract.start_date = self.currency_dates[0]
        except Exception as e:
            raise ValueError(f"Error getting the first date from the currency data: {e}")
        return first_contract.start_date
    
    def get_last_data_date(self):
        """
        Get the last date for the data of the future settlement.
        The date is in datetime format.
        """
        if len(self.contracts) == 0:
            raise ValueError(f"No contracts available for future {self.name}.")
        contracts_sorted = sorted(self.contracts, key=lambda x: x.last_trade_date)
        last_contract = contracts_sorted[-1]
        try:
            if self.currency_object is not None:
                if last_contract.last_trade_date > self.currency_dates[-1]:
                    last_contract.last_trade_date = self.currency_dates[-1]
        except Exception as e:
            raise ValueError(f"Error getting the last date from the currency data: {e}")
        return last_contract.last_trade_date
    
    def get_all_dates(self):
        """
        Get all dates for the data of the future settlement.
        The date is in datetime format.
        """
        if len(self.contracts) == 0:
            raise ValueError(f"No contracts available for future {self.name}.")
        all_dates = pd.DatetimeIndex([])
        for contract in self.contracts:
            all_dates = all_dates.union(contract.PX_SETTLE.index)
        # TODO: This will be hardcoded for now, but we will change it later
        # We set a maximum date for all the futures
        # It is set to the 30th of April 2025
        max_date = pd.Timestamp('2025-04-08')
        all_dates = all_dates[all_dates <= max_date]
        all_dates = all_dates.drop_duplicates()
        return all_dates
    
    ###############################################################################
    
    # @njit(cache=True)
    def return_roll_settle(self, initial_investment: float = 1000, date_delta: int = 0, maturity_delta: int = 0, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None):
        """
        Return a time series of the value of a theoretical portfolio of initial_cash invested in the future, being rolled
        It can be rolled at the maturity date, or date_delta days before the maturity date.
        It is normally rolled always with the next maturity contract, but (although not recommended) it can be rolled with the second, third, etc. contract (this is done with maturity_delta).
        initial_cash: the initial amount of cash invested in the future
        date_delta: the number of days to add to the date. This is useful if we want to roll before the maturity date.
        maturity_delta: the number of contracts to roll. This is useful if we want to roll with the second, third, etc. contract.
        start_date: the start date of the time series. If None, it will be the first date of the future.
        end_date: the end date of the time series. If None, it will be the last date of the future.
        """
        if start_date is None:
            start_date = self.get_first_data_date()
        if end_date is None:
            end_date = self.get_last_data_date()
            
        # Checking that the dates are in datetime format
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        except ValueError:
            raise ValueError("Start date and end date must be in datetime format.")
        
        # CHecking if start date is before end date
        if start_date > end_date:
            raise ValueError("Start date must be before end date.")
        
        # Getting the dates
        dates = pd.to_datetime(self.get_all_dates())
        dates = dates[(dates >= start_date) & (dates <= end_date)]
        dates = dates.sort_values()
        
        # Checking if start date is the fist date
        if start_date != dates[0]:
            raise ValueError("Start date must be the first date of the future.")
        
        # Creating a dataframe to store the data
        roll_settle_theoretical = pd.DataFrame(index=dates, columns=['Roll value'])
        roll_settle_theoretical['Roll value'] = np.nan
        roll_settle_theoretical['Transaction'] = np.nan
        roll_settle_theoretical['Transaction (BOOLEAN)'] = np.nan
        roll_settle_theoretical['Active contract'] = np.nan
        roll_settle_theoretical['Number of contracts'] = np.nan
        roll_settle_theoretical['Contract PX_SETTLE'] = np.nan
        
        # Initializing the roll - for the starting date
        incoming_contract = self.get_relevant_contract(start_date, date_delta, maturity_delta)
        roll_settle_theoretical.loc[start_date, 'Roll value'] = initial_investment
        roll_settle_theoretical.loc[start_date, 'Transaction'] = 'Buy: ' + incoming_contract.name + ' at ' + str(incoming_contract.PX_SETTLE.loc[start_date]) + ', totalling ' + str(initial_investment) + ' ' + incoming_contract.currency
        roll_settle_theoretical.loc[start_date, 'Transaction (BOOLEAN)'] = True
        roll_settle_theoretical.loc[start_date, 'Active contract'] = incoming_contract.name
        roll_settle_theoretical.loc[start_date, 'Number of contracts'] = initial_investment / incoming_contract.PX_SETTLE.loc[start_date]
        roll_settle_theoretical.loc[start_date, 'Contract PX_SETTLE'] = incoming_contract.PX_SETTLE.loc[start_date]
        
        # Looping through the dates
        for date in dates[1:]:
            date_index = dates.get_loc(date)
            yesterday = dates[date_index - 1]
            
            # Checking two things: Todays PX_SETTLE and yesterday PX_SETTLE
            # We're checking that they are not NaN, not empty, and numeric. If today is NaN we will assign it to yesterday's value
            if pd.isna(self.get_relevant_contract(date, date_delta, maturity_delta).PX_SETTLE.loc[date]):
                try:
                    self.get_relevant_contract(date, date_delta, maturity_delta).PX_SETTLE.loc[date] = self.get_relevant_contract(yesterday, date_delta, maturity_delta).PX_SETTLE.loc[yesterday]
                except Exception as e:
                    raise ValueError(f"Error getting the PX_SETTLE data for date {date}: {e}")
            if pd.isna(self.get_relevant_contract(yesterday, date_delta, maturity_delta).PX_SETTLE.loc[yesterday]):
                try:
                    self.get_relevant_contract(yesterday, date_delta, maturity_delta).PX_SETTLE.loc[yesterday] = self.get_relevant_contract(date, date_delta, maturity_delta).PX_SETTLE.loc[date]
                except Exception as e:
                    raise ValueError(f"Error getting the PX_SETTLE data for date {yesterday}: {e}")
            if pd.isna(self.get_relevant_contract(yesterday, date_delta, maturity_delta).PX_SETTLE.loc[date]):
                try:
                    self.get_relevant_contract(yesterday, date_delta, maturity_delta).PX_SETTLE.loc[date] = self.get_relevant_contract(yesterday, date_delta, maturity_delta).PX_SETTLE.loc[yesterday]
                except Exception as e:
                    raise ValueError(f"Error getting the PX_SETTLE data for date {yesterday}: {e}")
            
            # Checking if we should roll:
            if self.is_relevant_maturity_date(date, date_delta, maturity_delta):
                # Outgoing contract information - Marked to market to today
                outgoing_contract = self.get_relevant_contract(yesterday, date_delta, maturity_delta)
                outgoing_contract_no = roll_settle_theoretical.loc[yesterday, 'Number of contracts']
                outgoing_contract_px = outgoing_contract.PX_SETTLE.loc[date]
                outgoing_contract_value = outgoing_contract_no * outgoing_contract_px
                
                # Incoming contract information - Marked to market to today
                incoming_contract = self.get_relevant_contract(date, date_delta, maturity_delta)
                incoming_contract_no = outgoing_contract_value / incoming_contract.PX_SETTLE.loc[date]
                incoming_contract_px = incoming_contract.PX_SETTLE.loc[date]
                incoming_contract_value = incoming_contract_no * incoming_contract_px
                
                # Updating the roll settle theoretical value
                roll_settle_theoretical.loc[date, 'Roll value'] = incoming_contract_value
                roll_settle_theoretical.loc[date, 'Transaction'] = 'Buy: ' + incoming_contract.name + ' at ' + str(round(incoming_contract.PX_SETTLE.loc[date], 2)) + ', totalling ' + str(round(incoming_contract_value, 2)) + ' ' + incoming_contract.currency
                roll_settle_theoretical.loc[date, 'Transaction'] += ' - Sell: ' + outgoing_contract.name + ' at ' + str(round(outgoing_contract.PX_SETTLE.loc[date], 2)) + ', totalling ' + str(round(outgoing_contract_value, 2)) + ' ' + outgoing_contract.currency
                roll_settle_theoretical.loc[date, 'Transaction (BOOLEAN)'] = True
                roll_settle_theoretical.loc[date, 'Active contract'] = incoming_contract.name
                roll_settle_theoretical.loc[date, 'Number of contracts'] = incoming_contract_no
                roll_settle_theoretical.loc[date, 'Contract PX_SETTLE'] = incoming_contract.PX_SETTLE.loc[date]
            else:
                # If not roll, we update
                active_contract = self.get_relevant_contract(date, date_delta, maturity_delta)
                active_contract_no = roll_settle_theoretical.loc[yesterday, 'Number of contracts']
                active_contract_px = active_contract.PX_SETTLE.loc[date]
                active_contract_value = active_contract_no * active_contract_px
                
                roll_settle_theoretical.loc[date, 'Roll value'] = active_contract_value
                roll_settle_theoretical.loc[date, 'Transaction (BOOLEAN)'] = False
                roll_settle_theoretical.loc[date, 'Active contract'] = active_contract.name
                roll_settle_theoretical.loc[date, 'Number of contracts'] = active_contract_no
                roll_settle_theoretical.loc[date, 'Contract PX_SETTLE'] = active_contract.PX_SETTLE.loc[date]
        
        return roll_settle_theoretical

    
    def build_theoretical_roll_settle(self, initial_investment: float = 1000, date_delta: int = 0, maturity_delta: int = 0, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None, sd_filter: int = 5):
        """
        This runs the return_roll_settle function, storing the result within the class
        Then also performs some calculations for returns and log returns, which are also stored within the class
        Build a time series of the value of a theoretical portfolio of initial_cash invested in the future, being rolled
        It can be rolled at the maturity date, or date_delta days before the maturity date.
        It is normally rolled always with the next maturity contract, but (although not recommended) it can be rolled with the second, third, etc. contract (this is done with maturity_delta).
        initial_cash: the initial amount of cash invested in the future
        date_delta: the number of days to add to the date. This is useful if we want to roll before the maturity date.
        maturity_delta: the number of contracts to roll. This is useful if we want to roll with the second, third, etc. contract.
        start_date: the start date of the time series. If None, it will be the first date of the future.
        end_date: the end date of the time series. If None, it will be the last date of the future.
        """
        # Checking that the sd_filter is a positive integer
        if not isinstance(sd_filter, int) or sd_filter < 0:
            raise ValueError("sd_filter must be a positive integer.")
                
        
        try:
            self.roll_settle_theoretical = self.return_roll_settle(initial_investment, date_delta, maturity_delta, start_date, end_date)
            returns = self.roll_settle_theoretical['Roll value'].pct_change()
            log_returns = np.log(self.roll_settle_theoretical['Roll value'] / self.roll_settle_theoretical['Roll value'].shift(1))
            self.roll_settle_theoretical_returns = returns
            self.roll_settle_theoretical_log_returns = log_returns
            
            # Filtering the returns and log returns using the sd_filter (number of standard deviations from the mean)
            sd_returns = returns.std()
            sg_log_returns = log_returns.std()
            ub_returns = returns.mean() + sd_filter * sd_returns
            lb_returns = returns.mean() - sd_filter * sd_returns
            ub_log_returns = log_returns.mean() + sd_filter * sg_log_returns
            lb_log_returns = log_returns.mean() - sd_filter * sg_log_returns
            returns_filtered = returns[(returns < ub_returns) & (returns > lb_returns)]
            log_returns_filtered = log_returns[(log_returns < ub_log_returns) & (log_returns > lb_log_returns)]
            
            # Calculating the realized volatility for the filtered returns and log returns
            realized_vol_roll_1MROLL = returns_filtered.rolling(window=21).std()
            realized_vol_roll_3MROLL = returns_filtered.rolling(window=63).std()
            realized_vol_roll_6MROLL = returns_filtered.rolling(window=126).std()
            realized_vol_roll_12MROLL = returns_filtered.rolling(window=252).std()
            realized_vol_roll_log_1MROLL = log_returns_filtered.rolling(window=21).std()
            realized_vol_roll_log_3MROLL = log_returns_filtered.rolling(window=63).std()
            realized_vol_roll_log_6MROLL = log_returns_filtered.rolling(window=126).std()
            realized_vol_roll_log_12MROLL = log_returns_filtered.rolling(window=252).std()

            # Calculating downside realized volatility for filtered returns
            realized_downside_vol_roll_1MROLL = returns_filtered.rolling(window=21).apply(lambda x: downside_volatility(x), raw=False)
            realized_downside_vol_roll_3MROLL = returns_filtered.rolling(window=63).apply(lambda x: downside_volatility(x), raw=False)
            realized_downside_vol_roll_6MROLL = returns_filtered.rolling(window=126).apply(lambda x: downside_volatility(x), raw=False)
            realized_downside_vol_roll_12MROLL = returns_filtered.rolling(window=252).apply(lambda x: downside_volatility(x), raw=False)

            # Calculating garch volatility
            garch_volatility_1MROLL = returns_filtered.rolling(window=21).apply(lambda x: garch(x), raw=False)
            garch_volatility_3MROLL = returns_filtered.rolling(window=63).apply(lambda x: garch(x), raw=False)
            garch_volatility_6MROLL = returns_filtered.rolling(window=126).apply(lambda x: garch(x), raw=False)
            garch_volatility_12MROLL = returns_filtered.rolling(window=252).apply(lambda x: garch(x), raw=False)
                        
            # For all the dates not in the filtered data, we set the realized volatility to the last value (or NaN if no value)
            realized_vol_roll_1MROLL = realized_vol_roll_1MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            realized_vol_roll_3MROLL = realized_vol_roll_3MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            realized_vol_roll_6MROLL = realized_vol_roll_6MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            realized_vol_roll_12MROLL = realized_vol_roll_12MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            garch_volatility_1MROLL = garch_volatility_1MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            garch_volatility_3MROLL = garch_volatility_3MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            garch_volatility_6MROLL = garch_volatility_6MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            garch_volatility_12MROLL = garch_volatility_12MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            realized_vol_roll_log_1MROLL = realized_vol_roll_log_1MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            realized_vol_roll_log_3MROLL = realized_vol_roll_log_3MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            realized_vol_roll_log_6MROLL = realized_vol_roll_log_6MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            realized_vol_roll_log_12MROLL = realized_vol_roll_log_12MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            
            # Storing the realized volatility in the class
            self.realized_vol_roll_1MROLL = realized_vol_roll_1MROLL
            self.realized_vol_roll_3MROLL = realized_vol_roll_3MROLL
            self.realized_vol_roll_6MROLL = realized_vol_roll_6MROLL
            self.realized_vol_roll_12MROLL = realized_vol_roll_12MROLL
            self.realized_vol_roll_log_1MROLL = realized_vol_roll_log_1MROLL
            self.realized_vol_roll_log_3MROLL = realized_vol_roll_log_3MROLL
            self.realized_vol_roll_log_6MROLL = realized_vol_roll_log_6MROLL
            self.realized_vol_roll_log_12MROLL = realized_vol_roll_log_12MROLL
            self.realized_downside_vol_roll_1MROLL = realized_downside_vol_roll_1MROLL
            self.realized_downside_vol_roll_3MROLL = realized_downside_vol_roll_3MROLL
            self.realized_downside_vol_roll_6MROLL = realized_downside_vol_roll_6MROLL
            self.realized_downside_vol_roll_12MROLL = realized_downside_vol_roll_12MROLL
            self.garch_volatility_1MROLL = garch_volatility_1MROLL
            self.garch_volatility_3MROLL = garch_volatility_3MROLL
            self.garch_volatility_6MROLL = garch_volatility_6MROLL
            self.garch_volatility_12MROLL = garch_volatility_12MROLL
        
            # If the currency object is not None, we will also build the realized volatility for the fx rate returns
            # And the net realized volatility for the future (we first transform the price to the base currency, then we calculate the returns)
            if self.currency_object is not None:
                # We get all the dates get_all_dates
                # And if there are dates for which px_last is not available for those dates
                # We will interpolate the px_last to get the values for those dates
                # Getting the dates
                # dates = pd.to_datetime(self.get_all_dates())
                # dates = dates[(dates >= start_date) & (dates <= end_date)]
                # dates = dates.sort_values()
                
                # Getting the currency data for the dates
                currency_data = self.currency_object.px_last.loc[self.roll_settle_theoretical.index]
                currency_data = currency_data.dropna()
                currency_data = currency_data[~currency_data.index.isna()]
                currency_data = currency_data[~currency_data.index.isin([pd.NaT])]
                currency_data = currency_data[~currency_data.index.duplicated(keep='first')]
                currency_data = currency_data.reindex(self.roll_settle_theoretical.index, method='ffill')
                
                # Getting the currency returns and log returns
                currency_returns = currency_data.pct_change()
                currency_log_returns = np.log(currency_data / currency_data.shift(1))
                
                # Filtering the returns and log returns using the sd_filter (number of standard deviations from the mean)
                sd_currency_returns = currency_returns.std()
                sd_currency_log_returns = currency_log_returns.std()
                ub_currency_returns = currency_returns.mean() + sd_filter * sd_currency_returns
                lb_currency_returns = currency_returns.mean() - sd_filter * sd_currency_returns
                ub_currency_log_returns = currency_log_returns.mean() + sd_filter * sd_currency_log_returns
                lb_currency_log_returns = currency_log_returns.mean() - sd_filter * sd_currency_log_returns
                currency_returns_filtered = currency_returns[(currency_returns < ub_currency_returns) & (currency_returns > lb_currency_returns)]
                currency_log_returns_filtered = currency_log_returns[(currency_log_returns < ub_currency_log_returns) & (currency_log_returns > lb_currency_log_returns)]
                
                # Calculating the realized volatility for the filtered returns and log returns
                realized_vol_currency_1MROLL = currency_returns_filtered.rolling(window=21).std()
                realized_vol_currency_3MROLL = currency_returns_filtered.rolling(window=63).std()
                realized_vol_currency_6MROLL = currency_returns_filtered.rolling(window=126).std()
                realized_vol_currency_12MROLL = currency_returns_filtered.rolling(window=252).std()
                realized_vol_currency_log_1MROLL = currency_log_returns_filtered.rolling(window=21).std()
                realized_vol_currency_log_3MROLL = currency_log_returns_filtered.rolling(window=63).std()
                realized_vol_currency_log_6MROLL = currency_log_returns_filtered.rolling(window=126).std()
                realized_vol_currency_log_12MROLL = currency_log_returns_filtered.rolling(window=252).std()

                # Downside realized volatility for filtered returns
                realized_downside_vol_currency_1MROLL = currency_returns_filtered.rolling(window=21).apply(lambda x: downside_volatility(x), raw=False)
                realized_downside_vol_currency_3MROLL = currency_returns_filtered.rolling(window=63).apply(lambda x: downside_volatility(x), raw=False)
                realized_downside_vol_currency_6MROLL = currency_returns_filtered.rolling(window=126).apply(lambda x: downside_volatility(x), raw=False)
                realized_downside_vol_currency_12MROLL = currency_returns_filtered.rolling(window=252).apply(lambda x: downside_volatility(x), raw=False)

                # GARCH Volatility Estimation
                garch_volatility_currency_1MROLL = currency_returns_filtered.rolling(window=21).apply(lambda x: garch(x), raw=False)
                garch_volatility_currency_3MROLL = currency_returns_filtered.rolling(window=63).apply(lambda x: garch(x), raw=False)
                garch_volatility_currency_6MROLL = currency_returns_filtered.rolling(window=126).apply(lambda x: garch(x), raw=False)
                garch_volatility_currency_12MROLL = currency_returns_filtered.rolling(window=252).apply(lambda x: garch(x), raw=False)
                
                # For all the dates not in the filtered data, we set the realized volatility to the last value (or NaN if no value)
                realized_vol_currency_1MROLL = realized_vol_currency_1MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_vol_currency_3MROLL = realized_vol_currency_3MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_vol_currency_6MROLL = realized_vol_currency_6MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_vol_currency_12MROLL = realized_vol_currency_12MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_vol_currency_log_1MROLL = realized_vol_currency_log_1MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_vol_currency_log_3MROLL = realized_vol_currency_log_3MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_vol_currency_log_6MROLL = realized_vol_currency_log_6MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_vol_currency_log_12MROLL = realized_vol_currency_log_12MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                garch_volatility_currency_1MROLL = garch_volatility_currency_1MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                garch_volatility_currency_3MROLL = garch_volatility_currency_3MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                garch_volatility_currency_6MROLL = garch_volatility_currency_6MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                garch_volatility_currency_12MROLL= garch_volatility_currency_12MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_downside_vol_currency_1MROLL = realized_downside_vol_currency_1MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_downside_vol_currency_3MROLL = realized_downside_vol_currency_3MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_downside_vol_currency_6MROLL = realized_downside_vol_currency_6MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_downside_vol_currency_12MROLL = realized_downside_vol_currency_12MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                
                # Storing the realized volatility in the class
                self.realized_vol_currency_1MROLL = realized_vol_currency_1MROLL
                self.realized_vol_currency_3MROLL = realized_vol_currency_3MROLL
                self.realized_vol_currency_6MROLL = realized_vol_currency_6MROLL
                self.realized_vol_currency_12MROLL = realized_vol_currency_12MROLL
                self.realized_vol_currency_log_1MROLL = realized_vol_currency_log_1MROLL
                self.realized_vol_currency_log_3MROLL = realized_vol_currency_log_3MROLL
                self.realized_vol_currency_log_6MROLL = realized_vol_currency_log_6MROLL
                self.realized_vol_currency_log_12MROLL = realized_vol_currency_log_12MROLL
                self.realized_downside_vol_currency_1MROLL = realized_downside_vol_currency_1MROLL
                self.realized_downside_vol_currency_3MROLL = realized_downside_vol_currency_3MROLL
                self.realized_downside_vol_currency_6MROLL = realized_downside_vol_currency_6MROLL
                self.realized_downside_vol_currency_12MROLL = realized_downside_vol_currency_12MROLL
                self.garch_volatility_currency_1MROLL = garch_volatility_currency_1MROLL
                self.garch_volatility_currency_3MROLL = garch_volatility_currency_3MROLL
                self.garch_volatility_currency_6MROLL = garch_volatility_currency_6MROLL
                self.garch_volatility_currency_12MROLL = garch_volatility_currency_12MROLL

                
                # Calculating the net realized volatility for the future
                # We first transform the price to the base currency, then we calculate the returns
                # Getting the price in the base currency
                # Align indices before multiplying
                fx_px_last_aligned = self.currency_object.px_last.reindex(self.roll_settle_theoretical.index, method='ffill')
                price_base_currency = self.roll_settle_theoretical['Roll value'] * fx_px_last_aligned
                price_base_currency = price_base_currency.dropna()
                price_base_currency = price_base_currency[~price_base_currency.index.isna()]
                price_base_currency = price_base_currency[~price_base_currency.index.isin([pd.NaT])]
                price_base_currency = price_base_currency[~price_base_currency.index.duplicated(keep='first')]
                price_base_currency = price_base_currency.reindex(self.roll_settle_theoretical.index, method='ffill')
                
                # Getting the returns and log returns
                price_base_currency_returns = price_base_currency.pct_change()
                price_base_currency_log_returns = np.log(price_base_currency / price_base_currency.shift(1))
                
                # Filtering the returns and log returns using the sd_filter (number of standard deviations from the mean)
                sd_price_base_currency_returns = price_base_currency_returns.std()
                sd_price_base_currency_log_returns = price_base_currency_log_returns.std()
                ub_price_base_currency_returns = price_base_currency_returns.mean() + sd_filter * sd_price_base_currency_returns
                lb_price_base_currency_returns = price_base_currency_returns.mean() - sd_filter * sd_price_base_currency_returns
                ub_price_base_currency_log_returns = price_base_currency_log_returns.mean() + sd_filter * sd_price_base_currency_log_returns
                lb_price_base_currency_log_returns = price_base_currency_log_returns.mean() - sd_filter * sd_price_base_currency_log_returns
                price_base_currency_returns_filtered = price_base_currency_returns[(price_base_currency_returns < ub_price_base_currency_returns) & (price_base_currency_returns > lb_price_base_currency_returns)]
                price_base_currency_log_returns_filtered = price_base_currency_log_returns[(price_base_currency_log_returns < ub_price_base_currency_log_returns) & (price_base_currency_log_returns > lb_price_base_currency_log_returns)]
                
                # Calculating the realized volatility for the filtered returns and log returns
                realized_vol_price_base_currency_1MROLL = price_base_currency_returns_filtered.rolling(window=21).std()
                realized_vol_price_base_currency_3MROLL = price_base_currency_returns_filtered.rolling(window=63).std()
                realized_vol_price_base_currency_6MROLL = price_base_currency_returns_filtered.rolling(window=126).std()
                realized_vol_price_base_currency_12MROLL = price_base_currency_returns_filtered.rolling(window=252).std()
                realized_vol_price_base_currency_log_1MROLL = price_base_currency_log_returns_filtered.rolling(window=21).std()
                realized_vol_price_base_currency_log_3MROLL = price_base_currency_log_returns_filtered.rolling(window=63).std()
                realized_vol_price_base_currency_log_6MROLL = price_base_currency_log_returns_filtered.rolling(window=126).std()
                realized_vol_price_base_currency_log_12MROLL = price_base_currency_log_returns_filtered.rolling(window=252).std()

                # Downside realized volatility for filtered returns
                realized_downside_vol_price_base_currency_1MROLL = price_base_currency_returns_filtered.rolling(window=21).apply(lambda x: downside_volatility(x), raw=False)
                realized_downside_vol_price_base_currency_3MROLL = price_base_currency_returns_filtered.rolling(window=63).apply(lambda x: downside_volatility(x), raw=False)
                realized_downside_vol_price_base_currency_6MROLL = price_base_currency_returns_filtered.rolling(window=126).apply(lambda x: downside_volatility(x), raw=False)
                realized_downside_vol_price_base_currency_12MROLL = price_base_currency_returns_filtered.rolling(window=252).apply(lambda x: downside_volatility(x), raw=False)
                
                # For all the dates not in the filtered data, we set the realized volatility to the last value (or NaN if no value)
                realized_vol_price_base_currency_1MROLL = realized_vol_price_base_currency_1MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_vol_price_base_currency_3MROLL = realized_vol_price_base_currency_3MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_vol_price_base_currency_6MROLL = realized_vol_price_base_currency_6MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_vol_price_base_currency_12MROLL = realized_vol_price_base_currency_12MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_downside_vol_price_base_currency_1MROLL = realized_downside_vol_price_base_currency_1MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_downside_vol_price_base_currency_3MROLL = realized_downside_vol_price_base_currency_3MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_downside_vol_price_base_currency_6MROLL = realized_downside_vol_price_base_currency_6MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_downside_vol_price_base_currency_12MROLL = realized_downside_vol_price_base_currency_12MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_vol_price_base_currency_log_1MROLL = realized_vol_price_base_currency_log_1MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_vol_price_base_currency_log_3MROLL = realized_vol_price_base_currency_log_3MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_vol_price_base_currency_log_6MROLL = realized_vol_price_base_currency_log_6MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                realized_vol_price_base_currency_log_12MROLL = realized_vol_price_base_currency_log_12MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
                
                # Storing the realized volatility in the class
                self.realized_vol_price_base_currency_1MROLL = realized_vol_price_base_currency_1MROLL
                self.realized_vol_price_base_currency_3MROLL = realized_vol_price_base_currency_3MROLL
                self.realized_vol_price_base_currency_6MROLL = realized_vol_price_base_currency_6MROLL
                self.realized_vol_price_base_currency_12MROLL = realized_vol_price_base_currency_12MROLL
                self.realized_vol_price_base_currency_log_1MROLL = realized_vol_price_base_currency_log_1MROLL
                self.realized_vol_price_base_currency_log_3MROLL = realized_vol_price_base_currency_log_3MROLL
                self.realized_vol_price_base_currency_log_6MROLL = realized_vol_price_base_currency_log_6MROLL
                self.realized_vol_price_base_currency_log_12MROLL = realized_vol_price_base_currency_log_12MROLL
                self.realized_downside_vol_price_base_currency_1MROLL = realized_downside_vol_price_base_currency_1MROLL
                self.realized_downside_vol_price_base_currency_3MROLL = realized_downside_vol_price_base_currency_3MROLL
                self.realized_downside_vol_price_base_currency_6MROLL = realized_downside_vol_price_base_currency_6MROLL
                self.realized_downside_vol_price_base_currency_12MROLL = realized_downside_vol_price_base_currency_12MROLL
            
            # We check all the voatilities we have built
            # It can never go to 0. If it does we have to do a forward fill
            for vol in ['realized_vol_roll_1MROLL', 'realized_vol_roll_3MROLL', 'realized_vol_roll_6MROLL', 'realized_vol_roll_12MROLL', 'realized_vol_roll_log_1MROLL', 'realized_vol_roll_log_3MROLL', 'realized_vol_roll_log_6MROLL', 'realized_vol_roll_log_12MROLL']:
                if getattr(self, vol).isna().sum() > 0:
                    getattr(self, vol).fillna(method='ffill', inplace=True)
                    
                # Now any 0 values we have to set to the last value
                if (getattr(self, vol) == 0).sum() > 0:
                    getattr(self, vol).replace(0, np.nan, inplace=True)
                    getattr(self, vol).fillna(method='ffill', inplace=True)
            
            # The same but now if the currency object is not None
            if self.currency_object is not None:
                for vol in ['realized_vol_currency_1MROLL', 'realized_vol_currency_3MROLL', 'realized_vol_currency_6MROLL', 'realized_vol_currency_12MROLL', 'realized_vol_currency_log_1MROLL', 'realized_vol_currency_log_3MROLL', 'realized_vol_currency_log_6MROLL', 'realized_vol_currency_log_12MROLL']:
                    if getattr(self, vol).isna().sum() > 0:
                        getattr(self, vol).fillna(method='ffill', inplace=True)
                        
                    # Now any 0 values we have to set to the last value
                    if (getattr(self, vol) == 0).sum() > 0:
                        getattr(self, vol).replace(0, np.nan, inplace=True)
                        getattr(self, vol).fillna(method='ffill', inplace=True)
                
                # The same but now for the price base currency
                for vol in ['realized_vol_price_base_currency_1MROLL', 'realized_vol_price_base_currency_3MROLL', 'realized_vol_price_base_currency_6MROLL', 'realized_vol_price_base_currency_12MROLL', 'realized_vol_price_base_currency_log_1MROLL', 'realized_vol_price_base_currency_log_3MROLL', 'realized_vol_price_base_currency_log_6MROLL', 'realized_vol_price_base_currency_log_12MROLL']:
                    if getattr(self, vol).isna().sum() > 0:
                        getattr(self, vol).fillna(method='ffill', inplace=True)
                        
                    # Now any 0 values we have to set to the last value
                    if (getattr(self, vol) == 0).sum() > 0:
                        getattr(self, vol).replace(0, np.nan, inplace=True)
                        getattr(self, vol).fillna(method='ffill', inplace=True)
            
        except ValueError:
            raise ValueError(f"Error in return_roll_settle: {self.name} - {self.type} - {self.currency}")
        
    def build_realized_vol_undr(self, sd_filter: int = 5):
        """
        Build a realized volatility for the underlying.
        The date is in datetime format.
        """
        if self.underlying_data is None:
            raise ValueError("Underlying data is not available for this future.")
        if not isinstance(sd_filter, int) or sd_filter < 0:
            raise ValueError("sd_filter must be a positive integer.")
        
        try:
            underlying_returns = self.underlying_data.pct_change()
            underlying_log_returns = np.log(self.underlying_data / self.underlying_data.shift(1))
            
            # Filtering the returns and log returns using the sd_filter (number of standard deviations from the mean)
            sd_returns = underlying_returns.std()
            sd_log_returns = underlying_log_returns.std()
            ub_returns = underlying_returns.mean() + sd_filter * sd_returns
            lb_returns = underlying_returns.mean() - sd_filter * sd_returns
            ub_log_returns = underlying_log_returns.mean() + sd_filter * sd_log_returns
            lb_log_returns = underlying_log_returns.mean() - sd_filter * sd_log_returns
            underlying_returns_filtered = underlying_returns[(underlying_returns < ub_returns) & (underlying_returns > lb_returns)]
            underlying_log_returns_filtered = underlying_log_returns[(underlying_log_returns < ub_log_returns) & (underlying_log_returns > lb_log_returns)]
            
            # Calculating the realized volatility for the filtered returns and log returns
            realized_vol_undr_1MROLL = underlying_returns_filtered.rolling(window=21).std()
            realized_vol_undr_3MROLL = underlying_returns_filtered.rolling(window=63).std()
            realized_vol_undr_6MROLL = underlying_returns_filtered.rolling(window=126).std()
            realized_vol_undr_12MROLL = underlying_returns_filtered.rolling(window=252).std()
            realized_vol_undr_log_1MROLL = underlying_log_returns_filtered.rolling(window=21).std()
            realized_vol_undr_log_3MROLL = underlying_log_returns_filtered.rolling(window=63).std()
            realized_vol_undr_log_6MROLL = underlying_log_returns_filtered.rolling(window=126).std()
            realized_vol_undr_log_12MROLL = underlying_log_returns_filtered.rolling(window=252).std()
            
            # For all the dates not in the filtered data, we set the realized volatility to the last value (or NaN if no value)
            realized_vol_undr_1MROLL = realized_vol_undr_1MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            realized_vol_undr_3MROLL = realized_vol_undr_3MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            realized_vol_undr_6MROLL = realized_vol_undr_6MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            realized_vol_undr_12MROLL = realized_vol_undr_12MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            realized_vol_undr_log_1MROLL = realized_vol_undr_log_1MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            realized_vol_undr_log_3MROLL = realized_vol_undr_log_3MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            realized_vol_undr_log_6MROLL = realized_vol_undr_log_6MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')
            realized_vol_undr_log_12MROLL = realized_vol_undr_log_12MROLL.reindex(self.roll_settle_theoretical.index, method='ffill')

            # Storing the realized volatility in the class
            self.realized_vol_undr_1MROLL = realized_vol_undr_1MROLL
            self.realized_vol_undr_3MROLL = realized_vol_undr_3MROLL
            self.realized_vol_undr_6MROLL = realized_vol_undr_6MROLL
            self.realized_vol_undr_12MROLL = realized_vol_undr_12MROLL
            self.realized_vol_undr_log_1MROLL = realized_vol_undr_log_1MROLL
            self.realized_vol_undr_log_3MROLL = realized_vol_undr_log_3MROLL
            self.realized_vol_undr_log_6MROLL = realized_vol_undr_log_6MROLL
            self.realized_vol_undr_log_12MROLL = realized_vol_undr_log_12MROLL
        except ValueError:
            raise ValueError(f"Error in build_realized_vol_undr: {self.name} - {self.type} - {self.currency}")



class Strategy:
    """A class that represents a portfolio construction strategy using futures contracts."""
    def __init__(self, name: str, futures: list, strategy_type: str, base_currency: str, initial_investment: float = 1000, date_delta: int = 0, maturity_delta: int = 0, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None, sd_filter: int = 5):
        self.name = name
        self.futures = futures
        self.strategy_type = strategy_type
        self.base_currency = base_currency
        self.initial_investment = initial_investment
        self.date_delta = date_delta
        self.maturity_delta = maturity_delta
        self.start_date = start_date
        self.end_date = end_date
        self.sd_filter = sd_filter
        
        self.multiple_currency = False
        self.multiple_asset_class = False     
        
        # Checking the input types
        if not isinstance(self.name, str):
            raise TypeError("Strategy name must be a string.")
        if not isinstance(self.futures, list):
            raise TypeError("Futures must be a list of Future objects.")
        if not all(isinstance(fut, Future) for fut in self.futures):
            raise TypeError("All elements in 'futures' must be instances of the 'future' class.")
        if not isinstance(self.strategy_type, str):
            raise TypeError("Strategy type must be a string.")
        if self.strategy_type not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Strategy type {self.strategy_type} not supported. Supported types are: {list(SUPPORTED_STRATEGIES.keys())}.")
        if not isinstance(self.base_currency, str):
            raise TypeError("Base currency must be a string.")
        if self.base_currency not in SUPPORTED_CURRENCIES:
            raise ValueError(f"Base currency {self.base_currency} not supported. Supported currencies are: {list(SUPPORTED_CURRENCIES.keys())}.")
        if not isinstance(self.initial_investment, (int, float)):
            raise TypeError("Initial investment must be a number.")
        if not isinstance(self.date_delta, int):
            raise TypeError("Date delta must be an integer.")
        if not isinstance(self.maturity_delta, int):
            raise TypeError("Maturity delta must be an integer.")
        if self.date_delta < 0:
            raise ValueError("Date delta must be positive.")
        if self.maturity_delta < 0:
            raise ValueError("Maturity delta must be positive.")
        if not isinstance(self.start_date, (pd.Timestamp, type(None))):
            raise TypeError("Start date must be a pandas Timestamp or None.")
        if not isinstance(self.end_date, (pd.Timestamp, type(None))):
            raise TypeError("End date must be a pandas Timestamp or None.")
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError("Start date must be before end date.")
        if not isinstance(self.sd_filter, int):
            raise TypeError("SD filter must be a positive integer.")
        if self.sd_filter < 0:
            raise ValueError("SD filter must be a positive integer.")
        
        
        # Checking if within all the futures, there are multiple currencies and multiple asset classes
        currencies = set()
        asset_classes = set()
        for future in self.futures:
            if future.currency is not None:
                currencies.add(future.currency)
            if future.type is not None:
                asset_classes.add(future.type)
        if len(currencies) > 1:
            self.multiple_currency = True
        if len(asset_classes) > 1:
            self.multiple_asset_class = True
            
        # Controlling start_date and end_date
        first_common_date = [future.get_first_data_date() for future in self.futures]
        last_common_date = [future.get_last_data_date() for future in self.futures]
        first_common_date = max(first_common_date)
        last_common_date = min(last_common_date)
        
        # If start_date is None, we set it to the first common date
        # And if start_date is before the first common date, we set it to the first common date
        if self.start_date is None:
            self.start_date = first_common_date
        else:
            if self.start_date < first_common_date:
                self.start_date = first_common_date
        
        # If end_date is None, we set it to the last common date
        # And if end_date is after the last common date, we set it to the last common date
        if self.end_date is None:
            self.end_date = last_common_date
        else:
            if self.end_date > last_common_date:
                self.end_date = last_common_date
        
    #############################################################################
    
    def __repr__(self):
        return f"Strategy: {self.name}, Futures: {self.futures}, Strategy Type: {self.strategy_type}, Initial Investment: {self.initial_investment}, Date Delta: {self.date_delta}, Start Date: {self.start_date}, End Date: {self.end_date}"
    def __str__(self):
        return f"Strategy({self.name}, for {self.futures}, with {self.strategy_type} strategy)"
    def __hash__(self):
        return hash(self.name)
    
    ################################################################################
    
    def get_all_dates(self):
        """
        Get all dates for the data of the future settlement.
        The date is in datetime format.
        """
        all_dates = pd.DatetimeIndex([])  # Start with an empty DatetimeIndex
        for future in self.futures:
            all_dates = all_dates.union(future.get_all_dates())  # Union with each future's dates
        all_dates = all_dates.drop_duplicates()  # Ensure uniqueness
        return all_dates
    
    def build_rebalancing_calendar(self, rebalance_frequency: str = '3M', start_date: pd.Timestamp = None, end_date: pd.Timestamp = None):
        """
        Build a rebalancing calendar for the strategy.
        """
        # Making sure the rebalance frequency is valid
        if rebalance_frequency not in SUPPORTED_REBAL_FREQS:
            raise ValueError(f"Rebalance frequency {rebalance_frequency} not supported. Supported frequencies are: {list(SUPPORTED_REBAL_FREQS.keys())}.")
        
        # Building the rebalancing calendar
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
            
        # Getting the rebalancing dates
        # Rebalancing at business month start
        # 1M = 1 month, 3M = 3 months, 6M = 6 months, 12M = 12 months
        if rebalance_frequency == '1M':
            rebalancing_dates = pd.date_range(start=start_date, end=end_date, freq='BMS')
        elif rebalance_frequency == '3M':
            rebalancing_dates = pd.date_range(start=start_date, end=end_date, freq='3BMS')
        elif rebalance_frequency == '6M':
            rebalancing_dates = pd.date_range(start=start_date, end=end_date, freq='6BMS')
        elif rebalance_frequency == '12M':
            rebalancing_dates = pd.date_range(start=start_date, end=end_date, freq='12BMS')
        
        # Theoretically the next code is not needed, but just in case:
        # Making sure that the rebalancing dates are in the dates of the futures
        # If not, we will rebalance in the next available date within the futures
        # Example: Theoretically the 3M rebalance hits on a Sunday, but the future data is only available on working days, so we will rebalance on the next available date
        dates_futures = []
        for future in self.futures:
            dates_futures += list(future.get_all_dates())
        dates_futures = list(set(dates_futures))
        dates_futures.sort()
        rebalancing_dates = pd.to_datetime(rebalancing_dates)
        for date in rebalancing_dates:
            if date not in dates_futures:
                if [dates_futures] == []:
                    raise ValueError(f"Rebalancing date {date} is not in the future dates.")
                elif [d for d in dates_futures if d > date] == []:
                    pass
                else:
                    # If the date is not in the futures, we will rebalance in the next available date within the futures
                    next_date = min([d for d in dates_futures if d > date])
                    rebalancing_dates[rebalancing_dates == date] = next_date
        
        # Making sure that the rebalancing dates are unique
        rebalancing_dates = pd.Series(rebalancing_dates).drop_duplicates()
        rebalancing_dates = pd.to_datetime(rebalancing_dates)
        rebalancing_dates = rebalancing_dates[(rebalancing_dates >= start_date) & (rebalancing_dates <= end_date)]
        rebalancing_dates = rebalancing_dates.sort_values()
        return rebalancing_dates
    
    def build_weights(self, strategy_type: str, date: pd.Timestamp, estimation_window: str = '1M'):
        """
        Build the weights according to the strategy type.
        Since some strategies are time dependent, we need to pass the date.
        As more strategies are implemented, this function will be more complex.
        """
        # Checking the strategy type
        if strategy_type not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Strategy type {strategy_type} not supported. Supported types are: {list(SUPPORTED_STRATEGIES.keys())}.")
        # If the strategy type is 'EQUAL_WEIGHTED', we don't care about the date or the estimation window
        if strategy_type == 'EQUAL_WEIGHTED':
            pass
        else:
            if estimation_window not in SUPPORTED_ESTIMATION_WINDOWS:
                raise ValueError(f"Estimation window {estimation_window} not supported. Supported windows are: {list(SUPPORTED_ESTIMATION_WINDOWS.keys())}.")
        
        # Building the weights
        if strategy_type == 'EQUAL_WEIGHTED':
            weights = {}
            weight = 1 / len(self.futures)
            for future in self.futures:
                weights[future.name] = weight
        elif strategy_type == 'INVERSE_VARIANCE':
            weights = {}
            for future in self.futures:
                # Getting the realized volatility for the future
                # Which is already calculated in the future class
                # But it has to match the estimation window
                realized_vol = None
                if future.currency_object is None:
                    try:
                        if estimation_window == '1M':
                            realized_vol = future.realized_vol_roll_1MROLL.loc[date]
                            if realized_vol == 0:
                                # We try to get the former value
                                date_index = future.realized_vol_roll_1MROLL.index
                                yesterday = date_index[date_index.get_loc(date) - 1]
                                realized_vol = future.realized_vol_roll_1MROLL.loc[yesterday]
                        elif estimation_window == '3M':
                            realized_vol = future.realized_vol_roll_3MROLL.loc[date]
                            if realized_vol == 0:
                                # We try to get the former value
                                date_index = future.realized_vol_roll_3MROLL.index
                                yesterday = date_index[date_index.get_loc(date) - 1]
                                realized_vol = future.realized_vol_roll_3MROLL.loc[yesterday]
                        elif estimation_window == '6M':
                            realized_vol = future.realized_vol_roll_6MROLL.loc[date]
                            if realized_vol == 0:
                                # We try to get the former value
                                date_index = future.realized_vol_roll_6MROLL.index
                                yesterday = date_index[date_index.get_loc(date) - 1]
                                realized_vol = future.realized_vol_roll_6MROLL.loc[yesterday]
                        elif estimation_window == '12M':
                            realized_vol = future.realized_vol_roll_12MROLL.loc[date]
                            if realized_vol == 0:
                                # We try to get the former value
                                date_index = future.realized_vol_roll_12MROLL.index
                                yesterday = date_index[date_index.get_loc(date) - 1]
                                realized_vol = future.realized_vol_roll_12MROLL.loc[yesterday]
                    except KeyError:
                        raise KeyError(f"Realized volatility for future {future.name} not available for date {date}.")
                    if realized_vol is None:
                        raise ValueError(f"Realized volatility for future {future.name} is None for date {date}.")
                    if realized_vol == 0:
                        raise ValueError(f"Realized volatility for future {future.name} is 0 for date {date}.")
                    if pd.isna(realized_vol):
                        raise ValueError(f"Realized volatility for future {future.name} is NaN for date {date}.")
                    # The weight is the inverse of the realized volatility squared
                    weights[future.name] = 1 / (realized_vol ** 2)
                    if weights[future.name] == np.nan:
                        raise ValueError(f"Weight for future {future.name} is NaN.")
                else: # We will have to take the base currency vol and not the normal realized vol
                    try:
                        if estimation_window == '1M':
                            realized_vol = future.realized_vol_currency_1MROLL.loc[date]
                            if realized_vol == 0:
                                # We try to get the former value
                                date_index = future.realized_vol_currency_1MROLL.index
                                yesterday = date_index[date_index.get_loc(date) - 1]
                                realized_vol = future.realized_vol_currency_1MROLL.loc[yesterday]
                        elif estimation_window == '3M':
                            realized_vol = future.realized_vol_currency_3MROLL.loc[date]
                            if realized_vol == 0:
                                # We try to get the former value
                                date_index = future.realized_vol_currency_3MROLL.index
                                yesterday = date_index[date_index.get_loc(date) - 1]
                                realized_vol = future.realized_vol_currency_3MROLL.loc[yesterday]
                        elif estimation_window == '6M':
                            realized_vol = future.realized_vol_currency_6MROLL.loc[date]
                            if realized_vol == 0:
                                # We try to get the former value
                                date_index = future.realized_vol_currency_6MROLL.index
                                yesterday = date_index[date_index.get_loc(date) - 1]
                                realized_vol = future.realized_vol_currency_6MROLL.loc[yesterday]
                        elif estimation_window == '12M':
                            realized_vol = future.realized_vol_currency_12MROLL.loc[date]
                            if realized_vol == 0:
                                # We try to get the former value
                                date_index = future.realized_vol_currency_12MROLL.index
                                yesterday = date_index[date_index.get_loc(date) - 1]
                                realized_vol = future.realized_vol_currency_12MROLL.loc[yesterday]
                    except KeyError:
                        raise KeyError(f"Realized volatility for future {future.name} not available for date {date}.")
                    if realized_vol is None:
                        raise ValueError(f"Realized volatility for future {future.name} is None for date {date}.")
                    if realized_vol == 0:
                        raise ValueError(f"Realized volatility for future {future.name} is 0 for date {date}.")
                    if pd.isna(realized_vol):
                        raise ValueError(f"Realized volatility for future {future.name} is NaN for date {date}.")
                    # The weight is
                    # the inverse of the realized volatility squared
                    weights[future.name] = 1 / (realized_vol ** 2)
                    if weights[future.name] == np.nan:
                        raise ValueError(f"Weight for future {future.name} is NaN.")
            # Normalizing the weights
            # total_weight = sum(weights.values())
            #for future in self.futures:
            #    weights[future.name] = weights[future.name] / total_weight
        else:
            raise NotImplementedError(f"Strategy type {strategy_type} not implemented yet.")
        
        # Checking if the final weights are valid
        for future in self.futures:
            if weights[future.name] is None:
                raise ValueError(f"Weight for future {future.name} is None.")
            
        for future in self.futures:
            if weights[future.name] == np.nan:
                raise ValueError(f"Weight for future {future.name} is NaN.")
        
        # Checking if the weights sum to 1
        total_weight = 0
        for future in self.futures:
            if weights[future.name] is None:
                raise ValueError(f"Weight for future {future.name} is None.")
            if weights[future.name] == np.nan:
                raise ValueError(f"Weight for future {future.name} is NaN.")
            total_weight += weights[future.name]
        if total_weight != 1:
            # We normalize the weights
            for future in self.futures:
                weights[future.name] = weights[future.name] / total_weight

        total_weight = 0
        for future in self.futures:
            if weights[future.name] is None:
                raise ValueError(f"Weight for future {future.name} is None.")
            if weights[future.name] == np.nan:
                raise ValueError(f"Weight for future {future.name} is NaN.")
            total_weight += weights[future.name]
        if not np.isclose(total_weight, 1.0, atol=1e-8):
            raise ValueError(f"Weights do not sum to 1. Total weight: {total_weight}.")
        return weights
    
    # @njit(cache=True)
    def simulate_strategy_theoretical(self, strategy_type: str, initial_investment: float = 1000, date_delta: int = 0, maturity_delta: int = 0, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None, rebalance_frequency: str = '3M', estimation_window: str = '1M'):
        """
        Simulate the strategy, by rolling the futures and applying the specified portfolio construction strategy.
        The simulation is done in a theoretical way, by rolling using only PX_SETTLE.
        This assumes that no transaction costs are incurred in the rolling process, or anytime a transaction is made.
        """
        # Checking the strategy type
        if strategy_type not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Strategy type {strategy_type} not supported. Supported types are: {list(SUPPORTED_STRATEGIES.keys())}.")
        # Checking the rebalance frequency
        if rebalance_frequency not in SUPPORTED_REBAL_FREQS:
            raise ValueError(f"Rebalance frequency {rebalance_frequency} not supported. Supported frequencies are: {list(SUPPORTED_REBAL_FREQS.keys())}.")
        # Checking the estimation window
        if estimation_window not in SUPPORTED_ESTIMATION_WINDOWS:
            raise ValueError(f"Estimation window {estimation_window} not supported. Supported windows are: {list(SUPPORTED_ESTIMATION_WINDOWS.keys())}.")
        
        # Setting the dates if not provided
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
            
        # Checking that the dates are in datetime format
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        except ValueError:
            raise ValueError("Start date and end date must be in datetime format.")
        
        # Checking if start date is before end date
        if start_date > end_date:
            raise ValueError("Start date must be before end date.")
        
        # Getting the dates
        dates = pd.to_datetime(self.get_all_dates())
        dates = dates[(dates >= start_date) & (dates <= end_date)]
        dates = dates.sort_values()
        
        if strategy_type != 'EQUAL_WEIGHTED':
            # To check this, we will have to loop through the dates
            # Try building the weights
            # We check for the first date that the summ of all weights is 1
            # That is the first_variance_date for us
            # So we stop the loop when we find the first date that the sum of all weights is 1
            first_variance_date = None
            for date in dates:
                try:
                    weights = self.build_weights(strategy_type, date, estimation_window)
                    try:
                        total_weight = sum(weights.values())
                    except:
                        total_weight = 0
                    if total_weight == 1:
                        first_variance_date = date
                        break
                except:
                    # If we get an error, we just continue
                    pass
            
            # If first_variance_date is None
            # We raise an error saying that there's a problem with the weights
            if first_variance_date is None:
                raise ValueError(f"Could not find a date where the sum of weights is 1 for strategy {strategy_type}.")
            
            if start_date < first_variance_date:
                    start_date = first_variance_date
                    
            # Getting the dates again
            dates = pd.to_datetime(self.get_all_dates())
            dates = dates[(dates >= start_date) & (dates <= end_date)]
            dates = dates.sort_values()
            
        # Building the rebalancing calendar
        rebalancing_dates = self.build_rebalancing_calendar(rebalance_frequency, start_date, end_date)
        rebalancing_dates = rebalancing_dates[(rebalancing_dates >= start_date) & (rebalancing_dates <= end_date)]
        rebalancing_dates = rebalancing_dates.sort_values()
        rebalancing_dates = pd.DatetimeIndex(rebalancing_dates)
        
        # If multiple currencies we will bring the currency object of one of the foreign currency futures
        # TODO: If this is developed further, we need to do something more complex to accound for multiple FX rates
        if self.multiple_currency:
            # We will take the first future that has a currency object
            for future in self.futures:
                if future.currency_object is not None:
                    currency_object = future.currency_object
                    
                    # We will filter the the dates of the currency object to match our dates
                    # And if there's some NaN values we will forward fill them
                    currency_object.px_last = currency_object.px_last.reindex(dates, method='ffill')
                    currency_object.px_bid = currency_object.px_bid.reindex(dates, method='ffill')
                    currency_object.px_ask = currency_object.px_ask.reindex(dates, method='ffill')
                
                    # We check if the direction of the currency is correct
                    if currency_object.quote_currency != 'EUR':
                        raise ValueError(f"Currency object {currency_object.name} has a quote currency different from EUR.")   
                    break             
        
        # Creating an object to store the data
        strategy_simulation = {}
        strategy_simulation['Rebalancing calendar'] = rebalancing_dates
        strategy_simulation['Portfolio'] = pd.DataFrame(index=dates, columns=['Portfolio value'])
        strategy_simulation['Portfolio']['Portfolio value'] = np.nan
        strategy_simulation['Portfolio']['Transaction (BOOLEAN)'] = np.nan
        strategy_simulation['Portfolio']['Rolling (BOOLEAN)'] = np.nan
        strategy_simulation['Portfolio']['Rebalancing (BOOLEAN)'] = np.nan
        
        # For weights
        strategy_simulation['Weights'] = pd.DataFrame(index=dates, columns=[future.name for future in self.futures])
        for column in strategy_simulation['Weights'].columns:
            strategy_simulation['Weights'][column] = np.nan

        # Now for each future
        for future in self.futures:
            strategy_simulation[future.name] = pd.DataFrame(index=dates, columns=['Position value'])
            strategy_simulation[future.name]['Position value'] = np.nan
            strategy_simulation[future.name]['Active contract'] = np.nan
            strategy_simulation[future.name]['Number of contracts'] = np.nan
            strategy_simulation[future.name]['Contract PX_SETTLE'] = np.nan
            strategy_simulation[future.name]['Transaction'] = np.nan
            strategy_simulation[future.name]['Transaction (BOOLEAN)'] = np.nan
            strategy_simulation[future.name]['Rolling (BOOLEAN)'] = np.nan
            strategy_simulation[future.name]['Rebalancing (BOOLEAN)'] = np.nan
        
        # Initializing the portfolio - for the starting date
        weights = self.build_weights(strategy_type, start_date, estimation_window)
        strategy_simulation['Portfolio'].loc[start_date, 'Portfolio value'] = initial_investment
        strategy_simulation['Portfolio'].loc[start_date, 'Transaction (BOOLEAN)'] = True
        strategy_simulation['Portfolio'].loc[start_date, 'Rolling (BOOLEAN)'] = False
        strategy_simulation['Portfolio'].loc[start_date, 'Rebalancing (BOOLEAN)'] = True
        
        # Saving some simulation parameters
        strategy_simulation['Parameters'] = pd.DataFrame(index=[start_date], columns=['Initial investment', 'Date delta', 'Maturity delta', 'Start date', 'End date', 'Rebalance frequency', 'Estimation window', 'Strategy type'])
        strategy_simulation['Parameters']['Initial investment'] = initial_investment
        strategy_simulation['Parameters']['Date delta'] = date_delta
        strategy_simulation['Parameters']['Maturity delta'] = maturity_delta
        strategy_simulation['Parameters']['Start date'] = start_date
        strategy_simulation['Parameters']['End date'] = end_date
        strategy_simulation['Parameters']['Rebalance frequency'] = rebalance_frequency
        strategy_simulation['Parameters']['Estimation window'] = estimation_window
        strategy_simulation['Parameters']['Strategy type'] = strategy_type
        
        # Saving the weights
        for future in self.futures:
            strategy_simulation['Weights'].loc[start_date, future.name] = weights[future.name]
        
        for future in self.futures:
            strategy_simulation[future.name].loc[start_date, 'Position value'] = initial_investment * weights[future.name]
            strategy_simulation[future.name].loc[start_date, 'Transaction (BOOLEAN)'] = True
            strategy_simulation[future.name].loc[start_date, 'Rolling (BOOLEAN)'] = False
            strategy_simulation[future.name].loc[start_date, 'Rebalancing (BOOLEAN)'] = True
            if future.currency_object is not None:
                # Try with start_date FX first
                try:
                    fx_value = currency_object.px_last.loc[start_date]
                except KeyError:
                    fx_value = np.nan
                try:
                    px_settle = future.get_relevant_contract(start_date, date_delta, maturity_delta).PX_SETTLE.loc[start_date]
                except KeyError:
                    px_settle = np.nan
                contract_px_settle = px_settle * fx_value
                # If NaN, try next days for FX
                if pd.isna(contract_px_settle):
                    # We will backward fill the FX value and PX_SETTLE until we find a value for both
                    for i in range(1, 10):
                        try:
                            fx_value = currency_object.px_last.loc[dates[i-1]]
                            px_settle = future.get_relevant_contract(dates[i-1], date_delta, maturity_delta).PX_SETTLE.loc[dates[i-1]]
                            contract_px_settle = px_settle * fx_value
                            if not pd.isna(contract_px_settle):
                                break
                        except KeyError:
                            raise KeyError(f"Contract {future.name} not available for date {start_date}.")
                    
                strategy_simulation[future.name].loc[start_date, 'Contract PX_SETTLE'] = contract_px_settle
            else:
                strategy_simulation[future.name].loc[start_date, 'Contract PX_SETTLE'] = future.get_relevant_contract(start_date, date_delta, maturity_delta).PX_SETTLE.loc[start_date]
            strategy_simulation[future.name].loc[start_date, 'Active contract'] = future.get_relevant_contract(start_date, date_delta, maturity_delta).name
            strategy_simulation[future.name].loc[start_date, 'Number of contracts'] = initial_investment * weights[future.name] / strategy_simulation[future.name].loc[start_date, 'Contract PX_SETTLE']
            strategy_simulation[future.name].loc[start_date, 'Transaction'] = f'Buy: {future.get_relevant_contract(start_date, date_delta, maturity_delta).name} at {future.get_relevant_contract(start_date, date_delta, maturity_delta).PX_SETTLE.loc[start_date]:.2f}  {future.get_relevant_contract(start_date, date_delta, maturity_delta).currency}, totalling {initial_investment * weights[future.name]:.2f} EUR'
            
            # Checking if nothing is NaN
            

            
        # Looping through the dates
        for date in dates[1:]:
            date_index = dates.get_loc(date)
            yesterday = dates[date_index - 1]
            tomorrow = None
            try:
                tomorrow = dates[date_index + 1]
            except IndexError:
                pass
            
            # Valuing the portfolio [Today's prices, yesterday's contracts, before any possible transaction is made]
            portfolio_value = 0
            for future in self.futures:
                # Getting the active contract at the close of the previous day
                active_contract = future.get_relevant_contract(yesterday, date_delta, maturity_delta)
                active_contract_no = strategy_simulation[future.name].loc[yesterday, 'Number of contracts']
                if future.currency_object is not None:
                    active_contract_px = active_contract.PX_SETTLE.loc[date] * currency_object.px_last.loc[date]
                else:
                    active_contract_px = active_contract.PX_SETTLE.loc[date]
                active_contract_value = active_contract_no * active_contract_px
                portfolio_value += active_contract_value
            
            # Updating the portfolio value. As of now, no transaction has been made
            strategy_simulation['Portfolio'].loc[date, 'Portfolio value'] = portfolio_value
            strategy_simulation['Portfolio'].loc[date, 'Transaction (BOOLEAN)'] = False
            strategy_simulation['Portfolio'].loc[date, 'Rolling (BOOLEAN)'] = False
            strategy_simulation['Portfolio'].loc[date, 'Rebalancing (BOOLEAN)'] = False
            
            # Now we go through each contract. First we revalue the contracts, then we check if we need to roll (and if we do, we roll).
            for future in self.futures:
                # Valuing contract: Asuming that yesterday's contract is the same as today's contract
                active_contract = future.get_relevant_contract(yesterday, date_delta, maturity_delta)
                active_contract_no = strategy_simulation[future.name].loc[yesterday, 'Number of contracts']
                if future.currency_object is not None:
                    active_contract_px = active_contract.PX_SETTLE.loc[date] * currency_object.px_last.loc[date]
                else:
                    active_contract_px = active_contract.PX_SETTLE.loc[date]
                active_contract_value = active_contract_no * active_contract_px
                strategy_simulation[future.name].loc[date, 'Position value'] = active_contract_value
                strategy_simulation[future.name].loc[date, 'Transaction (BOOLEAN)'] = False
                strategy_simulation[future.name].loc[date, 'Rolling (BOOLEAN)'] = False
                strategy_simulation[future.name].loc[date, 'Rebalancing (BOOLEAN)'] = False
                strategy_simulation[future.name].loc[date, 'Active contract'] = active_contract.name
                strategy_simulation[future.name].loc[date, 'Number of contracts'] = active_contract_no
                strategy_simulation[future.name].loc[date, 'Contract PX_SETTLE'] = active_contract_px
                strategy_simulation[future.name].loc[date, 'Transaction'] = None
                
                # Now we check if we need to roll
                if future.is_relevant_maturity_date(date, date_delta, maturity_delta):
                    # We get the incoming contract (if is_relevant_maturity_date is True, it means that today we need to roll)
                    incoming_contract = future.get_relevant_contract(date, date_delta, maturity_delta)
                    if future.currency_object is not None:
                        incoming_contract_px = incoming_contract.PX_SETTLE.loc[date] * currency_object.px_last.loc[date]
                    else:
                        incoming_contract_px = incoming_contract.PX_SETTLE.loc[date]
                    incoming_contract_no = strategy_simulation[future.name].loc[date, 'Position value'] / incoming_contract_px
                    strategy_simulation[future.name].loc[date, 'Active contract'] = incoming_contract.name
                    strategy_simulation[future.name].loc[date, 'Number of contracts'] = incoming_contract_no
                    strategy_simulation[future.name].loc[date, 'Contract PX_SETTLE'] = incoming_contract_px
                    strategy_simulation[future.name].loc[date, 'Transaction (BOOLEAN)'] = True
                    strategy_simulation[future.name].loc[date, 'Rolling (BOOLEAN)'] = True
                    strategy_simulation[future.name].loc[date, 'Rebalancing (BOOLEAN)'] = False
                    strategy_simulation[future.name].loc[date, 'Transaction'] = f'Buy: {incoming_contract.name} at {incoming_contract_px:.2f} EUR, totalling {strategy_simulation[future.name].loc[date, "Position value"]:.2f} EUR, Sell: {active_contract.name} at {active_contract_px:.2f} EUR, totalling {strategy_simulation[future.name].loc[yesterday, "Position value"]:.2f} EUR'
                    
                    # Updating booleans in the portfolio
                    strategy_simulation['Portfolio'].loc[date, 'Transaction (BOOLEAN)'] = True
                    strategy_simulation['Portfolio'].loc[date, 'Rolling (BOOLEAN)'] = True
                    
            # Now we check if we need to rebalance
            if pd.Timestamp(date) in rebalancing_dates or date.date() in rebalancing_dates.date:
                # We get the weights for the strategy
                try:
                    weights = self.build_weights(strategy_type, date, estimation_window)
                except ValueError:
                    # If realized volatility is not available for today, which is the ValueError we are raising in the build_weights function
                    # We will not rebalance today, and we will try to rebalance tomorrow
                    # So we change add tomorrow to the rebalancing dates
                    if tomorrow is not None:
                        rebalancing_dates = rebalancing_dates.append(pd.DatetimeIndex([tomorrow]))
                    continue
                
                # Rebalancing the portfolio
                for future in self.futures:
                    # Getting the active contract
                    old_active_contract = future.get_relevant_contract(yesterday, date_delta, maturity_delta)
                    active_contract = future.get_relevant_contract(date, date_delta, maturity_delta)
                    if future.currency_object is not None:
                        active_contract_px = active_contract.PX_SETTLE.loc[date] * currency_object.px_last.loc[date]
                    else:
                        active_contract_px = active_contract.PX_SETTLE.loc[date]
                    active_contract_no = strategy_simulation[future.name].loc[date, 'Number of contracts']
                    active_contract_value = active_contract_no * active_contract_px
                    
                    # Getting the new number of contracts
                    old_active_contract_no = strategy_simulation[future.name].loc[date, 'Number of contracts']
                    new_active_contract_no = strategy_simulation['Portfolio'].loc[date, 'Portfolio value'] * weights[future.name] / active_contract_px
                    if future.currency_object is not None:
                        old_active_contract_value = old_active_contract_no * old_active_contract.PX_SETTLE.loc[date] * currency_object.px_last.loc[date]
                    else:
                        old_active_contract_value = old_active_contract_no * old_active_contract.PX_SETTLE.loc[date]
                    new_active_contract_value = new_active_contract_no * active_contract_px
                    
                    # Updating the number of contracts
                    strategy_simulation[future.name].loc[date, 'Number of contracts'] = new_active_contract_no
                    strategy_simulation[future.name].loc[date, 'Transaction (BOOLEAN)'] = True
                    strategy_simulation[future.name].loc[date, 'Rolling (BOOLEAN)'] = False
                    strategy_simulation[future.name].loc[date, 'Rebalancing (BOOLEAN)'] = True
                    strategy_simulation[future.name].loc[date, 'Position value'] = new_active_contract_no * active_contract_px
                    strategy_simulation[future.name].loc[date, 'Active contract'] = active_contract.name
                    strategy_simulation[future.name].loc[date, 'Contract PX_SETTLE'] = active_contract_px
                    
                    # If the value of the position before rebalancing is greater than the value of the position after rebalancing, we need to sell the difference
                    # Otherwise, we need to buy the difference
                    # Except if it was also a roll date, in which case we sell all of the old contract and buy all of the new contract
                    if old_active_contract != active_contract:
                        # This means that we are rolling but also rebalancing
                        strategy_simulation[future.name].loc[date, 'Transaction'] = f'Buy: {active_contract.name} at {active_contract_px:.2f}, totalling {strategy_simulation['Portfolio'].loc[date, 'Portfolio value'] * weights[future.name]:.2f} EUR, Sell: {old_active_contract.name} at {active_contract_px:.2f}, totalling {old_active_contract_value:.2f} EUR'
                        strategy_simulation[future.name].loc[date, 'Rolling (BOOLEAN)'] = True
                        strategy_simulation['Portfolio'].loc[date, 'Rolling (BOOLEAN)'] = True
                    else:
                        strategy_simulation['Portfolio'].loc[date, 'Rolling (BOOLEAN)'] = False
                        if new_active_contract_value > active_contract_value:
                            # Amount to buy
                            amount_to_buy = new_active_contract_value - active_contract_value
                            contracts_to_buy = amount_to_buy / active_contract_px
                            strategy_simulation[future.name].loc[date, 'Transaction'] = f'Buy: {active_contract.name} at {active_contract_px:.2f}, totalling {amount_to_buy:.2f} EUR'
                        else:
                            # Amount to sell
                            amount_to_sell = active_contract_value - new_active_contract_value
                            contracts_to_sell = amount_to_sell / active_contract_px
                            strategy_simulation[future.name].loc[date, 'Transaction'] = f'Sell: {active_contract.name} at {active_contract_px:.2f}, totalling {amount_to_sell:.2f} EUR'
                    
                    strategy_simulation[future.name].loc[date, 'Transaction'] = f'Buy: {active_contract.name} at {active_contract_px:.2f}, totalling {strategy_simulation['Portfolio'].loc[date, 'Portfolio value'] * weights[future.name]:.2f} EUR, Sell: {active_contract.name} at {active_contract_px:.2f}, totalling {active_contract_value:.2f} EUR'
                    
                # Updating booleans in the portfolio
                strategy_simulation['Portfolio'].loc[date, 'Transaction (BOOLEAN)'] = True
                strategy_simulation['Portfolio'].loc[date, 'Rebalancing (BOOLEAN)'] = True

            # If something has gone wrong, by producing a NaN value
            # We will try to do a forward fill
            revalue = False
            for future in self.futures:
                # If any of these have occurred we will need to revalue the portfolio so that the weights sum to 1
                # Now checking if its nan
                if pd.isna(strategy_simulation[future.name].loc[date, 'Position value']):
                    print(f'Future {future.name} has a NaN value for date {date}. A forward fill will be applied.')
                    strategy_simulation[future.name].loc[date, 'Position value'] = strategy_simulation[future.name].loc[yesterday, 'Position value']
                    revalue = True
                if pd.isna(strategy_simulation[future.name].loc[date, 'Number of contracts']):
                    print(f'Future {future.name} has a NaN value for date {date}. A forward fill will be applied.')
                    strategy_simulation[future.name].loc[date, 'Number of contracts'] = strategy_simulation[future.name].loc[yesterday, 'Number of contracts']
                    revalue = True
                if pd.isna(strategy_simulation[future.name].loc[date, 'Contract PX_SETTLE']):
                    print(f'Future {future.name} has a NaN value for date {date}. A forward fill will be applied.')
                    strategy_simulation[future.name].loc[date, 'Contract PX_SETTLE'] = strategy_simulation[future.name].loc[yesterday, 'Contract PX_SETTLE']
                    revalue = True
            
            # If we need to revalue the portfolio, we will do it here
            if revalue:
                # Revaluing the portfolio
                portfolio_value = 0
                for future in self.futures:
                    # Getting the active contract at the close of the previous day
                    active_contract = future.get_relevant_contract(yesterday, date_delta, maturity_delta)
                    active_contract_no = strategy_simulation[future.name].loc[yesterday, 'Number of contracts']
                    if future.currency_object is not None:
                        active_contract_px = active_contract.PX_SETTLE.loc[date] * currency_object.px_last.loc[date]
                    else:
                        active_contract_px = active_contract.PX_SETTLE.loc[date]
                    active_contract_value = active_contract_no * active_contract_px
                    portfolio_value += active_contract_value
                
                # Updating the portfolio value. As of now, no transaction has been made
                strategy_simulation['Portfolio'].loc[date, 'Portfolio value'] = portfolio_value
                
            
            # Updating the weights
            for future in self.futures:
                try:
                    strategy_simulation['Weights'].loc[date, future.name] = strategy_simulation[future.name].loc[date, 'Position value'] / strategy_simulation['Portfolio'].loc[date, 'Portfolio value']
                except ZeroDivisionError:
                    strategy_simulation['Weights'].loc[date, future.name] = None
            
            # Checking if the weights sum to 1
            total_weight = 0
            for future in self.futures:
                if strategy_simulation['Weights'].loc[date, future.name] is None:
                    raise ValueError(f"Weight for future {future.name} is None.")
                if strategy_simulation['Weights'].loc[date, future.name] == np.nan:
                    raise ValueError(f"Weight for future {future.name} is NaN.")
                total_weight += strategy_simulation['Weights'].loc[date, future.name]
            if not np.isclose(total_weight, 1.0, atol=1e-8):
                raise ValueError(f"Weights do not sum to 1 at date {date}. Total weight: {total_weight}.")
        return strategy_simulation