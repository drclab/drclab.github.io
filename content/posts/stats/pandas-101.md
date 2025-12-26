+++
title = "Pandas 101"
date = "2025-12-26"
tags = ["pandas", "python", "data analysis"]
categories = ["posts"]
type = "post"
draft = false
math = true
description = "An introduction to the Pandas library for data analysis and manipulation using the World Happiness dataset."
+++

Welcome to Pandas 101! In this post, we will introduce the Pandas library, one of the most powerful tools for data analysis in Python. We will use the World Happiness Report dataset to demonstrate common operations like loading data, viewing data, selecting columns and rows, filtering, and basic plotting.

## 1. Importing the Libraries

The first step in any data analysis project is to import the necessary libraries. For this tutorial, we will need `pandas` for data manipulation and `seaborn` for additional visualization tools later on.

```python
import pandas as pd
import seaborn as sns
```

## 2. Importing the Data

We will use a CSV file containing data from the World Happiness Report. We can load this into a Pandas DataFrame using the `read_csv` function.

```python
# Import the data
df = pd.read_csv("world_happiness.csv")
```

## 3. Basic Operations With a Dataframe

### 3.1 View the Dataframe

Once the data is loaded, it's a good idea to take a quick look at it. You can use the `.head()` and `.tail()` methods to see the first and last few rows of the DataFrame.

```python
# Show the first 5 rows of the dataframe
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country name</th>
      <th>year</th>
      <th>Life Ladder</th>
      <th>Log GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy at birth</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
      <th>Positive affect</th>
      <th>Negative affect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2008</td>
      <td>3.724</td>
      <td>7.350</td>
      <td>0.451</td>
      <td>50.5</td>
      <td>0.718</td>
      <td>0.168</td>
      <td>0.882</td>
      <td>0.414</td>
      <td>0.258</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2009</td>
      <td>4.401</td>
      <td>7.508</td>
      <td>0.552</td>
      <td>50.8</td>
      <td>0.679</td>
      <td>0.191</td>
      <td>0.850</td>
      <td>0.481</td>
      <td>0.237</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>4.758</td>
      <td>7.613</td>
      <td>0.539</td>
      <td>51.1</td>
      <td>0.600</td>
      <td>0.123</td>
      <td>0.707</td>
      <td>0.517</td>
      <td>0.275</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2011</td>
      <td>3.832</td>
      <td>7.581</td>
      <td>0.521</td>
      <td>51.4</td>
      <td>0.496</td>
      <td>0.166</td>
      <td>0.731</td>
      <td>0.430</td>
      <td>0.267</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2012</td>
      <td>3.783</td>
      <td>7.661</td>
      <td>0.521</td>
      <td>51.7</td>
      <td>0.531</td>
      <td>0.238</td>
      <td>0.776</td>
      <td>0.614</td>
      <td>0.268</td>
    </tr>
  </tbody>
</table>
</div>

Similarly, we can view the last 5 rows, or specify the number of rows we want to see.

```python
# Show the last 2 rows of the dataframe
df.tail(2)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country name</th>
      <th>year</th>
      <th>Life Ladder</th>
      <th>Log GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy at birth</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
      <th>Positive affect</th>
      <th>Negative affect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2197</th>
      <td>Zimbabwe</td>
      <td>2021</td>
      <td>3.155</td>
      <td>7.657</td>
      <td>0.685</td>
      <td>54.050</td>
      <td>0.668</td>
      <td>-0.076</td>
      <td>0.757</td>
      <td>0.610</td>
      <td>0.242</td>
    </tr>
    <tr>
      <th>2198</th>
      <td>Zimbabwe</td>
      <td>2022</td>
      <td>3.296</td>
      <td>7.670</td>
      <td>0.666</td>
      <td>54.525</td>
      <td>0.652</td>
      <td>-0.070</td>
      <td>0.753</td>
      <td>0.641</td>
      <td>0.191</td>
    </tr>
  </tbody>
</table>
</div>

## 3.2 Index and Column Names

In a `DataFrame`, data is stored in a two-dimensional grid. The rows are indexed and the columns are named. You can access these using `df.index` and `df.columns`.

```python
df.index
```
`RangeIndex(start=0, stop=2199, step=1)`

```python
df.columns
```
`Index(['Country name', 'year', 'Life Ladder', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption', 'Positive affect', 'Negative affect'], dtype='object')`

### Renaming Columns

It is often useful to rename columns to remove spaces or make them more consistent. Here is a way to automatically replace spaces with underscores and convert names to lowercase:

```python
# Create a dictionary mapping old column names to new column names
columns_to_rename = {i: "_".join(i.split(" ")).lower() for i in df.columns}

# Rename the columns
df = df.rename(columns=columns_to_rename)
df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_name</th>
      <th>year</th>
      <th>life_ladder</th>
      <th>log_gdp_per_capita</th>
      <th>social_support</th>
      <th>healthy_life_expectancy_at_birth</th>
      <th>freedom_to_make_life_choices</th>
      <th>generosity</th>
      <th>perceptions_of_corruption</th>
      <th>positive_affect</th>
      <th>negative_affect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2008</td>
      <td>3.724</td>
      <td>7.350</td>
      <td>0.451</td>
      <td>50.5</td>
      <td>0.718</td>
      <td>0.168</td>
      <td>0.882</td>
      <td>0.414</td>
      <td>0.258</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2009</td>
      <td>4.402</td>
      <td>7.509</td>
      <td>0.552</td>
      <td>50.8</td>
      <td>0.679</td>
      <td>0.191</td>
      <td>0.850</td>
      <td>0.481</td>
      <td>0.237</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>4.758</td>
      <td>7.614</td>
      <td>0.539</td>
      <td>51.1</td>
      <td>0.600</td>
      <td>0.121</td>
      <td>0.707</td>
      <td>0.517</td>
      <td>0.275</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2011</td>
      <td>3.832</td>
      <td>7.581</td>
      <td>0.521</td>
      <td>51.4</td>
      <td>0.496</td>
      <td>0.164</td>
      <td>0.731</td>
      <td>0.480</td>
      <td>0.267</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2012</td>
      <td>3.783</td>
      <td>7.661</td>
      <td>0.521</td>
      <td>51.7</td>
      <td>0.531</td>
      <td>0.238</td>
      <td>0.776</td>
      <td>0.614</td>
      <td>0.268</td>
    </tr>
  </tbody>
</table>
</div>

## 3.3 Data Types

Each column in a Pandas DataFrame has a specific data type (`dtype`). This allows different types of data to coexist in the same table.

```python
df.dtypes
```

```text
country_name                         object
year                                  int64
life_ladder                         float64
log_gdp_per_capita                  float64
social_support                      float64
healthy_life_expectancy_at_birth    float64
freedom_to_make_life_choices        float64
generosity                          float64
perceptions_of_corruption           float64
positive_affect                     float64
negative_affect                     float64
dtype: object
```

You can also change the data types if necessary using `.astype()`:

```python
# Change the type of all float columns to float
float_columns = [i for i in df.columns if i not in ["country_name", "year"]]
df = df.astype({i: float for i in float_columns})
```

Finally, `df.info()` gives a concise summary of the DataFrame, including the number of non-null values.

```python
df.info()
```

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2199 entries, 0 to 2198
Data columns (total 11 columns):
 #   Column                            Non-Null Count  Dtype  
---  ------                            --------------  -----  
 0   country_name                      2199 non-null   object 
 1   year                              2199 non-null   int64  
 2   life_ladder                       2199 non-null   float64
 3   log_gdp_per_capita                2179 non-null   float64
 4   social_support                    2186 non-null   float64
 5   healthy_life_expectancy_at_birth  2145 non-null   float64
 6   freedom_to_make_life_choices      2166 non-null   float64
 7   generosity                        2126 non-null   float64
 8   perceptions_of_corruption         2083 non-null   float64
 9   positive_affect                   2175 non-null   float64
 10  negative_affect                   2183 non-null   float64
dtypes: float64(9), int64(1), object(1)
memory usage: 189.1+ KB
```

## 3.4 Selecting Columns

One way of selecting a single column is to use `df.column_name`. This returns a Pandas `Series`.

```python
# Select the life_ladder column
x = df.life_ladder
print(f"type(x):\n {type(x)}\n")
```

```text
type(x):
 <class 'pandas.core.series.Series'>
```

Another way is to use square brackets and the column name as a string:

```python
x = df["life_ladder"]
```

Passing a list of labels rather than a single label selects multiple columns and returns a DataFrame:

```python
# Selecting multiple columns
x = df[["life_ladder", "year"]]
```

## 3.5 Selecting Rows

You can use slicing to select a range of rows. This returns a DataFrame containing all columns for the specified rows.

```python
# Select rows 2, 3, and 4
df[2:5]
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_name</th>
      <th>year</th>
      <th>life_ladder</th>
      <th>log_gdp_per_capita</th>
      <th>social_support</th>
      <th>healthy_life_expectancy_at_birth</th>
      <th>freedom_to_make_life_choices</th>
      <th>generosity</th>
      <th>perceptions_of_corruption</th>
      <th>positive_affect</th>
      <th>negative_affect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>4.758</td>
      <td>7.614</td>
      <td>0.539</td>
      <td>51.1</td>
      <td>0.600</td>
      <td>0.121</td>
      <td>0.707</td>
      <td>0.517</td>
      <td>0.275</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2011</td>
      <td>3.832</td>
      <td>7.581</td>
      <td>0.521</td>
      <td>51.4</td>
      <td>0.496</td>
      <td>0.164</td>
      <td>0.731</td>
      <td>0.480</td>
      <td>0.267</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2012</td>
      <td>3.783</td>
      <td>7.661</td>
      <td>0.521</td>
      <td>51.7</td>
      <td>0.531</td>
      <td>0.238</td>
      <td>0.776</td>
      <td>0.614</td>
      <td>0.268</td>
    </tr>
  </tbody>
</table>
</div>

## 3.6 Iterating Over Rows

If you need to iterate through the data row by row, you can use `.iterrows()`. It returns an index and a Series for each row.

```python
index, row = next(df.iterrows())
print(row)
```

```text
country_name                        Afghanistan
year                                       2008
life_ladder                               3.724
log_gdp_per_capita                         7.35
social_support                            0.451
healthy_life_expectancy_at_birth           50.5
freedom_to_make_life_choices              0.718
generosity                                0.168
perceptions_of_corruption                 0.882
positive_affect                           0.414
negative_affect                           0.258
Name: 0, dtype: object
```

## 3.7 Boolean Indexing

Boolean indexing allows you to filter the DataFrame based on conditions. For example, selecting data only for the year 2022:

```python
# Select rows where the year is 2022
df_2022 = df[df["year"] == 2022]
```

After filtering, the index will still have its original values. You can reset the index using `.reset_index(drop=True)`:

```python
df_2022 = df_2022.reset_index(drop=True)
```

# 4. Summary Statistics

Pandas provides a quick way to calculate summary statistics for all numeric columns using `.describe()`.

```python
df.describe()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>life_ladder</th>
      <th>log_gdp_per_capita</th>
      <th>social_support</th>
      <th>healthy_life_expectancy_at_birth</th>
      <th>freedom_to_make_life_choices</th>
      <th>generosity</th>
      <th>perceptions_of_corruption</th>
      <th>positive_affect</th>
      <th>negative_affect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2199.000000</td>
      <td>2199.000000</td>
      <td>2179.000000</td>
      <td>2186.000000</td>
      <td>2145.000000</td>
      <td>2166.000000</td>
      <td>2126.000000</td>
      <td>2083.000000</td>
      <td>2175.000000</td>
      <td>2183.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2014.161437</td>
      <td>5.479227</td>
      <td>9.389760</td>
      <td>0.810681</td>
      <td>63.294582</td>
      <td>0.747847</td>
      <td>0.000091</td>
      <td>0.745208</td>
      <td>0.652148</td>
      <td>0.271493</td>
    </tr>
  </tbody>
</table>
</div>

# 5. Plotting

You can create basic plots directly from your DataFrame using `.plot()`.

### Basic Line Plot
By default, it plots all numeric columns against the index.

```python
df.plot()
```

![Line Plot](/images/pandas-101/line_plot.png)

### Scatter Plot
Scatter plots are useful for exploring relationships between variables. Here we plot Log GDP per capita vs Life Ladder score.

```python
df.plot(kind='scatter', x='log_gdp_per_capita', y='life_ladder')
```

![Scatter Plot](/images/pandas-101/scatter_plot.png)

### More Customization
You can also use custom colors and sizes:

```python
cmap = {'Brazil': 'Green', 'Slovenia': 'Orange', 'India': 'purple'}
df.plot(
    kind='scatter',
    x='log_gdp_per_capita',
    y='life_ladder',
    c=[cmap.get(c, 'yellow') for c in df.country_name],
    s=2
)
```

![Colored Scatter Plot](/images/pandas-101/scatter_plot_colored.png)

### Histogram
To see the distribution of a single column:

```python
df.hist("life_ladder")
```

![Histogram](/images/pandas-101/histogram.png)

### Pair Plot
With this kind of plot, you can see pairwise scatter plots for each pair of columns. On the diagonal (where both columns are the same), you don't have a scatter plot (which would only show a line), but a histogram showing the distribution of datapoints.

```python
sns.pairplot(df)
```

![Pair Plot](/images/pandas-101/pair_plot.png)

# 6. Operations on Columns

You can easily create new columns from existing ones using arithmetic operations.

```python
# Create a new column as a difference of two others
df["net_affect_difference"] = df["positive_affect"] - df["negative_affect"]
df.head()
```

### Applying Functions
For more advanced operations, use `.apply()`:

```python
# Rescale life_ladder using a lambda function
df['life_ladder_rescaled'] = df['life_ladder'].apply(lambda x: x / 10)

# Apply a custom function
def my_function(x):
    return x * 2

df['my_function'] = df['life_ladder'].apply(my_function)
```

**Congratulations!** You've completed Pandas 101. You now have the skills to start exploring and manipulating datasets using this powerful library.
