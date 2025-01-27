# -------------------------------------------------------------------------------------------
# This file contains a series of functions used for processing, analyzing, and visualizing data
# in the context of volatility managed portfolios. It also contains functions for portfolio
# construction, optimization, and backtesting - including adjustments for volatility management.
# This is intended as a module to be imported into other scripts or notebooks.
# This module is part of the Volatility Managed Portfolio project for EDHEC Business School.
# Nicolas Gamboa Alvarez, Wiktor Kotwicki, Moana Valdelaire, 2024 - 2025
# -------------------------------------------------------------------------------------------

# Importing necessary libraries -------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Functions ----------------------------------------------------------------------------------

# Function that takes a dataframe, optionally takes a column name. Then it should return the
# dataframe as a time series with the datetime column as index.
def to_time_series(df, col_name='Date', format='%Y-%m-%d'):
    """
    Converts a specified column in a DataFrame to datetime format and sets it as the index.
    Parameters:
    df (pandas.DataFrame): The DataFrame to convert.
    col_name (str): The name of the column to convert to datetime format. Default is 'Date'.
    format (str): The datetime format to use for conversion. Default is '%Y-%m-%d'.
    Returns:
    pandas.DataFrame: The DataFrame with the specified column converted to datetime format and set as the index.
    Raises:
    ValueError: If the specified column does not exist in the DataFrame or if the conversion to datetime fails.
    """
    
    if col_name in df.columns:
        try:
            df[col_name] = pd.to_datetime(df[col_name], format=format)
        except:
            raise ValueError('Error: Could not convert the column to datetime format.')
    else:
        raise ValueError('Error: The specified column does not exist in the DataFrame.')
    
    df.set_index(col_name, inplace=True)
    df.sort_index(inplace=True)
    return df


# Function that takes a dataframe which has a datetime index, and a column name
# Then it should build a plot of the time series of the column. 
# The plot should manage the formatting used for this project.
# TODO Advanced version: Should take as input the frequency of the analysis (hourly, daily, weekly, monthly)
# And the number of periods should be in years, but the function should be able to handle different frequencies.
def plot_time_series(df, target_var, title=None, xlabel=None, ylabel=None, figsize=(12, 6), color='darkblue', save=False, save_path=None, logscale=False, grid=False):
    """
    Plots a time series of a specified column in a DataFrame.
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the time series data.
    target_var (str): The name of the column to plot.
    title (str): The title of the plot. Default is None.
    xlabel (str): The label for the x-axis. Default is None.
    ylabel (str): The label for the y-axis. Default is None.
    figsize (tuple): The size of the plot. Default is (12, 6).
    Returns:
    None
    Raises:
    ValueError: If the specified column does not exist in the DataFrame.
    """
    
    if target_var in df.columns:
        plt.figure(figsize=figsize)
        if logscale:
            plt.yscale('log')
        plt.plot(df[target_var], color=color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(grid)
        plt.xlim(df.index[0], df.index[-1])
        if save:
            plt.savefig(save_path)
        plt.show()
        # Setting the x-axis min and max values to the first and last dates in the dataset

    else:
        raise ValueError('Error: The specified column does not exist in the DataFrame.')
        

# Function that takes a dataframe which has a datetime index, and a column name
# This function will assume that the column is a price time series and will calculate the returns
# Not annualized, and then plot the returns time series.
# The plot should manage the formatting used for this project.
# TODO Advanced version: Should take as input the frequency of the analysis (hourly, daily, weekly, monthly)
# And the number of periods should be in years, but the function should be able to handle different frequencies.
def plot_returns(df, target_var, title=None, xlabel=None, ylabel=None, figsize=(12, 6), color='darkred', save=False, save_path=None, logscale=False, grid=False):
    """
    Calculates the returns of a specified column in a DataFrame and plots the returns time series.
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the time series data.
    target_var (str): The name of the column to calculate returns and plot.
    title (str): The title of the plot. Default is None.
    xlabel (str): The label for the x-axis. Default is None.
    ylabel (str): The label for the y-axis. Default is None.
    figsize (tuple): The size of the plot. Default is (12, 6).
    Returns:
    None
    Raises:
    ValueError: If the specified column does not exist in the DataFrame.
    """
    
    if target_var in df.columns:
        df['Returns'] = df[target_var].pct_change()
        plt.figure(figsize=figsize)
        if logscale:
            plt.yscale('log')
        plt.plot(df['Returns'], color=color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(grid)
        plt.xlim(df.index[0], df.index[-1])
        if save:
            plt.savefig(save_path)
        plt.show()
    else:
        raise ValueError('Error: The specified column does not exist in the DataFrame.')

# Function that takes a dataframe which has a datetime index, and a column name
# This function will assume that the column is a price time series and will calculate a rolling
# volatility measure based on a window (ionput). The function should plot the volatility measure. 
# The plot should manage the formatting used for this project.
# TODO Advanced version: Should take as input the frequency of the analysis (hourly, daily, weekly, monthly)
# And the number of periods should be in years, but the function should be able to handle different frequencies.
def plot_rolling_volatility(df, target_var, window=252, title=None, xlabel=None, ylabel=None, figsize=(12, 6), color='darkgreen', save=False, save_path=None, logscale=False, grid=False):
    """
    Calculates the rolling volatility of a specified column in a DataFrame and plots the volatility measure.
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the time series data.
    target_var (str): The name of the column to calculate rolling volatility and plot.
    window (int): The window size for the rolling volatility calculation. Default is 252.
    title (str): The title of the plot. Default is None.
    xlabel (str): The label for the x-axis. Default is None.
    ylabel (str): The label for the y-axis. Default is None.
    figsize (tuple): The size of the plot. Default is (12, 6).
    Returns:
    None
    Raises:
    ValueError: If the specified column does not exist in the DataFrame.
    """
    
    if target_var in df.columns:
        df['Volatility'] = df[target_var].pct_change().rolling(window=window).std()
        plt.figure(figsize=figsize)
        if logscale:
            plt.yscale('log')
        plt.plot(df['Volatility'], color=color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(grid)
        plt.xlim(df.index[0], df.index[-1])
        if save:
            plt.savefig(save_path)
        plt.show()
    else:
        raise ValueError('Error: The specified column does not exist in the DataFrame.')

# Function that builds a vertically split graph, one for price time series and the other for volume time series.
# The x axis should be the same for both graphs. The function should manage the formatting used for this project.
# TODO Advanced version: Should take as input the frequency of the analysis (hourly, daily, weekly, monthly)
# And the number of periods should be in years, but the function should be able to handle different frequencies.
def plot_price_volume(df, target_price, target_volume, title=None, xlabel=None, ylabel_price=None, ylabel_volume=None, figsize=(12, 6), color_price='darkblue', color_volume='darkred', save=False, save_path=None, logscale=False, grid=False, ratio=0.7):
    """
    Plots a price time series and a volume time series in a vertically split graph.
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the time series data.
    target_price (str): The name of the column to plot as the price time series.
    target_volume (str): The name of the column to plot as the volume time series.
    title (str): The title of the plot. Default is None.
    xlabel (str): The label for the x-axis. Default is None.
    ylabel_price (str): The label for the y-axis of the price time series. Default is None.
    ylabel_volume (str): The label for the y-axis of the volume time series. Default is None.
    figsize (tuple): The size of the plot. Default is (12, 6).
    Returns:
    None
    Raises:
    ValueError: If the specified columns do not exist in the DataFrame.
    """
    
    # Making a two-panel plot with adjustable height ratio
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[ratio, 1-ratio])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Plotting the price time series
    ax1.plot(df[target_price], color=color_price)
    ax1.set_ylabel(ylabel_price)
    
    # Plotting the volume time series
    ax2.plot(df[target_volume], color=color_volume)
    ax2.set_ylabel(ylabel_volume)
    
    # Setting the title and x-axis label
    plt.suptitle(title)
    plt.xlabel(xlabel)
    
    # Setting the grid
    ax1.grid(grid)
    ax2.grid(grid)
    
    # Formatting the plot
    plt.xlim(df.index[0], df.index[-1])
    
    # Formatting the y-axis
    if logscale:
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    

    if save:
        plt.savefig(save_path)
    plt.show()
        
# Function that builds a vertically split graph, one for returns time series and the other for volatility time series.
# The x axis should be the same for both graphs. The function should manage the formatting used for this project.
# TODO Advanced version: Should take as input the frequency of the analysis (hourly, daily, weekly, monthly)
# And the number of periods should be in years, but the function should be able to handle different frequencies.
def plot_returns_volatility(df, target_returns, target_volatility, title=None, xlabel=None, ylabel_returns=None, ylabel_volatility=None, figsize=(12, 6), color_returns='darkred', color_volatility='darkgreen', save=False, save_path=None, logscale=False, grid=False, ratio=0.7, window=252):
    """
    Plots a returns time series and a volatility time series in a vertically split graph.
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the time series data.
    target_returns (str): The name of the column to plot as the returns time series.
    target_volatility (str): The name of the column to plot as the volatility time series.
    title (str): The title of the plot. Default is None.
    xlabel (str): The label for the x-axis. Default is None.
    ylabel_returns (str): The label for the y-axis of the returns time series. Default is None.
    ylabel_volatility (str): The label for the y-axis of the volatility time series. Default is None.
    figsize (tuple): The size of the plot. Default is (12, 6).
    Returns:
    None
    Raises:
    ValueError: If the specified columns do not exist in the DataFrame.
    """
    
    # Making a two-panel plot with adjustable height ratio
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[ratio, 1-ratio])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Generating the returns and volatility time series
    df['Returns'] = df[target_returns].pct_change()
    df['Volatility'] = df[target_returns].pct_change().rolling(window=window).std()
    
    # Plotting the returns time series
    ax1.plot(df['Returns'], color=color_returns)
    ax1.set_ylabel(ylabel_returns)
    
    # Plotting the volatility time series
    ax2.plot(df['Volatility'], color=color_volatility)
    ax2.set_ylabel(ylabel_volatility)
    
    # Setting the title and x-axis label
    plt.suptitle(title)
    plt.xlabel(xlabel)
    
    # Setting the grid
    ax1.grid(grid)
    ax2.grid(grid)
    
    # Formatting the plot
    plt.xlim(df.index[0], df.index[-1])
    
    # Formatting the y-axis
    if logscale:
        ax1.set_yscale('log')
        ax2.set_yscale('log')

    if save:
        plt.savefig(save_path)
    plt.show()

# Testing ------------------------------------------------------------------------------------

# Loading a sample dataset 'Kaggle_XAU_1d_data_2004_to_2024-09-20.csv'
df = pd.read_csv('Data/Kaggle_XAU_1d_data_2004_to_2024-09-20.csv')

# Displaying the first few rows of the dataset
print(df.head())

# Converting the 'Date' column to datetime format
df = to_time_series(df, format='%Y.%m.%d')

# Displaying the first few rows of the dataset after conversion
print(df.head())

# Plorring all functions
plot_time_series(df, 'Close', title='Gold Price Time Series', xlabel='Date', ylabel='Price', figsize=(10, 7), color='darkblue', save=True, save_path='gold_price.png', logscale=False, grid=False)
plot_returns(df, 'Close', title='Gold Returns Time Series', xlabel='Date', ylabel='Returns', figsize=(10, 7), color='darkred', save=True, save_path='gold_returns.png', logscale=False, grid=False)
plot_rolling_volatility(df, 'Close', title='Gold Volatility Time Series - 252 days rolling', xlabel='Date', ylabel='Volatility - 252 days rolling', figsize=(10, 7), color='darkgreen', save=True, save_path='gold_volatility.png', logscale=False, grid=False)
plot_price_volume(df, 'Close', 'Volume', title='Gold Price and Volume Time Series', xlabel='Date', ylabel_price='Price', ylabel_volume='Volume', figsize=(10, 7), color_price='darkblue', color_volume='darkred', save=True, save_path='gold_price_volume.png', logscale=False, grid=False, ratio=0.7)
plot_returns_volatility(df, 'Close', 'Close', title='Gold Returns and Volatility Time Series - Daily', xlabel='Date', ylabel_returns='Returns', ylabel_volatility='Volatility - 252 dyas rolling', figsize=(10, 7), color_returns='darkblue', color_volatility='darkred', save=True, save_path='gold_returns_volatility_daily.png', logscale=False, grid=False, ratio=0.7, window=252)