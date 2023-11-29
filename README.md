<a href="https://www.kaggle.com/code/jesusgraterol/bitcoin-price-state" target="_blank">
  <img align="left" alt="Kaggle" title="Open in Kaggle" src="./img/open-in-kaggle.svg">
</a><br>

# Bitcoin Price State

Understanding the price state of Bitcoin, stocks or any other asset is essential to open/increase/reduce/close positions as it can reveal the trend, as well as overbought and oversold states.

This notebook will go through an approach that provides a deep analysis of the price state for any asset within a time window. The state can then be used to generate events capable of triggering position actions at the "right" time.

```python
##################
## Dependencies ##
##################

from typing import Union, List, Tuple, Literal, TypedDict, Dict
from math import ceil
from random import randint
from numpy import mean, arange
from pandas import DataFrame, options
from datetime import datetime
import utilities as utils
import matplotlib.pyplot as plt
options.display.float_format = '{:.3f}'.format
```

```python
###########
## Types ##
###########


# Price State
# The price state of a window can be stateless (0) or stateful (-2|-1|1|2). 
# When a window is stateless, means the price has been moving sideways. On the
# other hand, when a state is detected, means the price is following a trend.
IState = Literal[-2, -1, 0, 1, 2]
IStateAlias = Literal["Decreasing Strongly", "Decreasing", "Stateless", "Increasing", "Increasing Strongly"]


# Split States
# In order to better analyze the state of a series of prices (candlesticks), 
# the sequence is split into different ranges, focusing more on the latest splits.
# The splits that are applied to the window sequence are the following:
# s100 -> 100% | s75 -> 75% | s50 - 50% | s25 -> 25%
# s15 -> 15% | s10 -> 10% | s5 -> 5% | s2 -> 2%


# The identifiers of the splits that will be applied to the window.
ISplitID = Literal["s100","s75","s50","s25","s15","s10","s5","s2"]

# The result of a split state calculation
class ISplitStateResult(TypedDict):
    # The state of the split
    state: IState
        
    # The percentual change in the split
    change: float
        
    # The payload used to calculate the result of the split
    payload: List[float]
        
# Full object containing all the split states & payloads
ISplitStates = Dict[ISplitID, ISplitStateResult]


# The final state dict for a window
class IWindowState(TypedDict):
    # The average state based on all the splits
    average_state: IState
        
    # The states by splits (payload included)
    split_states: ISplitStates
        
    # The price window (candlesticks)
    window: DataFrame
        

# Configuration
# The dictionary used to hold user input to prevent/limit the 
# editing of the source code.
class IConfig(TypedDict):
    candlesticks_interval: utils.IIntervalID
    window_width: int
    state_requirement: float
    strong_state_requirement: float
    samples_limit: int
```

```python
#############
## Globals ##
#############


# State Aliases
STATE_ALIAS: Dict[IState, IStateAlias] = {
    "-2": "Decreasing Strongly",
    "-1": "Decreasing",
     "0": "Stateless",
     "1": "Increasing",
     "2": "Increasing Strongly",
}
    
    
# Candlestick Chart Sizes
SMALL_FIG_SIZE: Tuple[int, int] = (6, 4)
```

## Configuration

```python
CONFIG: IConfig = {
    # The candlesticks interval that will be used in the dataset
    "candlesticks_interval": "15m",
    
    # The number of candlesticks that comprise a window
    "window_width": 128,
    
    # The percentage change requirements for a split to be stateful
    "state_requirement": 0.025,
    "strong_state_requirement": 0.85,
    
    # The number of samples that will be calculated and displayed
    "samples_limit": 50
}
```

## Dataset

```python
# Download the dataset for the given interval
ds: DataFrame = utils.get_historic_candlesticks("15m")
utils.plot_candlesticks(ds, display_volume=False, figsize=SMALL_FIG_SIZE)
ds.describe()
```
![image.png](./img/dataset-summary-01.png)
![image.png](./img/dataset-summary-02.png)

## Window

```python
def make_window(start_at_index: Union[None, int] = None) -> DataFrame:
    """Builds a window df based on the dataset. If no index is provided,
    it will generate a random one.
    
    Args:
        start_at_index: Union[None, int]
            The index at which the window should begin.
            
    Returns:
        DataFrame
    """
    # Random Index Generator
    def generate_random_index() -> int:
        return randint(0, ds.shape[0] - CONFIG["window_width"])
    
    # Initialize the start of the window
    start_at: int = start_at_index if isinstance(start_at_index, int) else generate_random_index()
    
    # Return the sliced df
    return ds.iloc[start_at: start_at + CONFIG["window_width"]]
```

```python
window: DataFrame = make_window()
utils.plot_candlesticks(make_window(), title="Window Sample", display_volume=False, figsize=SMALL_FIG_SIZE)
window.describe()
```
![image.png](./img/window-sample.png)

## State Calculator

```python
def calculate_average_state(states: List[IState]) -> IState:
    """Given a list of price states for all splits, it will calculate the 
    average and return it.
    
    Args:
        states: List[IState]
            The list of states by split. Order does not matter.
            
    Returns:
        IState
    """
    # Calculate the mean of all the states
    states_mean: float = mean(states)
        
    # Handle the final state accordingly
    if states_mean >= 1.5:
        return 2
    elif states_mean >= 0.75:
        return 1
    elif states_mean <= -1.5:
        return -2
    elif states_mean <= -0.75:
        return -1
    else:
        return 0
```

```python
def calculate_state_for_split(window_split: DataFrame) -> ISplitStateResult:
    """Based on the provided split, it will calculate the state and
    return the full dict with the result.
    
    Args:
        window_split: DataFrame
            The split's dataframe
    
    Returns:
        ISplitStateResult
    """
    # Initialize the split's list
    split_items: List[float] = [candlestick["c"] for candlestick in window_split.to_records()]
    
    # Calculate the % change between the first and last items
    change: float = utils.calculate_percentage_change(split_items[0], split_items[-1])
        
    # Calculate the split's state based on the % change
    state: IState = 0
    if change >= CONFIG["strong_state_requirement"]:
        state = 2
    elif change >= CONFIG["state_requirement"]:
        state = 1
    elif change <= -(CONFIG["strong_state_requirement"]):
        state = -2
    elif change <= -(CONFIG["state_requirement"]):
        state = -1
        
    # Finally, return the state
    return { "state": state, "change": change, "payload": split_items }
```

```python
def calculate_state(window: DataFrame) -> IWindowState:
    """Calculates the price state for a given window.
    
    Args:
        window: DataFrame
            The window in candlestick format.
            
    Returns:
        IWindowState
    """
    # Build the split states
    states: ISplitStates = {
        "s100": calculate_state_for_split(window),
        "s75": calculate_state_for_split(window.iloc[window.shape[0] - ceil(window.shape[0] * 0.75):]),
        "s50": calculate_state_for_split(window.iloc[window.shape[0] - ceil(window.shape[0] * 0.5):]),
        "s25": calculate_state_for_split(window.iloc[window.shape[0] - ceil(window.shape[0] * 0.25):]),
        "s15": calculate_state_for_split(window.iloc[window.shape[0] - ceil(window.shape[0] * 0.15):]),
        "s10": calculate_state_for_split(window.iloc[window.shape[0] - ceil(window.shape[0] * 0.10):]),
        "s5": calculate_state_for_split(window.iloc[window.shape[0] - ceil(window.shape[0] * 0.05):]),
        "s2": calculate_state_for_split(window.iloc[window.shape[0] - ceil(window.shape[0] * 0.02):]),
    }
        
    # Return the final build
    return {
        "average_state": calculate_average_state([
            states["s100"]["state"],
            states["s75"]["state"],
            states["s50"]["state"],
            states["s25"]["state"],
            states["s15"]["state"],
            states["s10"]["state"],
            states["s5"]["state"],
            states["s2"]["state"]
        ]),
        "split_states": states,
        "window": window
    }
```

## State Samples

```python
def display_state(window_state: IWindowState) -> None:
    """Displays the full window state object.
    
    Args:
        window_state: IWindowState
            The final state of the window
    """
    # Chart Misc Generators
    def get_chart_color(state: IState) -> str:
        if state == 2:
            return "darkgreen"
        elif state == 1:
            return "palegreen"
        elif state == -2:
            return "darkred"
        elif state == -1:
            return "lightcoral"
        else:
            return "grey"
    def get_chart_label(id: ISplitID, change: float) -> str:
        return f"{id}: {'+' if change > 0 else ''}{change}%"
    
    # Display the candlesticks chart
    utils.plot_candlesticks(
        window_state["window"], 
        title=f"State: {STATE_ALIAS[str(window_state['average_state'])]}", 
        display_volume=False, 
        figsize=(12, 4)
    )
    
    # Define subplots
    fig, ax = plt.subplots(2, 4, figsize=(10,3))
    fig.tight_layout()
    
    # Iterate over each split and plot its chart accordingly
    row: int = 0
    column: int = 0
    for split_id in ["s100","s75","s50","s25","s15","s10","s5","s2"]:
        # Plot the chart
        ax[row, column].plot(
            window_state["split_states"][split_id]["payload"], 
            color=get_chart_color(window_state["split_states"][split_id]["state"])
        )
        ax[row, column].set_title(get_chart_label(split_id, window_state["split_states"][split_id]["change"]))
        ax[row, column].axis("off")
        
        # Increment the row and|column accordingly
        if column >= 3:
            row += 1
            column = 0
        else:
            column += 1
```

```python
for _ in arange(CONFIG["samples_limit"]):
    display_state(calculate_state(make_window()))
```


![image.png](./img/samples/01.png)

---

![image.png](./img/samples/02.png)