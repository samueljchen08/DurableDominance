import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
file_path = 'Golf_Masters.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1', header=1)

# Clean the data by removing unnecessary columns and rows with NaN values
data_cleaned = data.drop(columns=['Unnamed: 0']).dropna()

# Filter out rows where Position contains 'Cut' or non-numeric values
data_cleaned = data_cleaned[~data_cleaned['Position'].str.contains('Cut', na=False)]

# Filter out rows where Final contains non-numeric values like 'E'
data_cleaned = data_cleaned[~data_cleaned['Final'].str.contains('E', na=False)]

# Convert relevant columns to numeric types
data_cleaned['Position'] = data_cleaned['Position'].replace({'T': ''}, regex=True).astype(int)
data_cleaned['Final'] = pd.to_numeric(data_cleaned['Final'])
data_cleaned['Year'] = data_cleaned['Year'].astype(int)

# Group by year and calculate the number of former finishes and standard deviation of scores
grouped = data_cleaned.groupby('Year').agg({
    'Player': 'count',
    'Final': 'std'
}).reset_index()

# Rename columns for clarity
grouped.columns = ['Year', 'Top_10_Count', 'Score_Std_Dev']

# Scatter plot with number of top 10 finishes on y-axis and standard deviation of scores on x-axis
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot
ax.scatter(grouped['Score_Std_Dev'], grouped['Top_10_Count'])

# Add labels and title
ax.set_xlabel('Standard Deviation of Scores')
ax.set_ylabel('Number of Top 10 Finishes')
ax.set_title('Number of Top 10 Finishes vs Standard Deviation of Scores by Year')

# Annotate each point with the corresponding year
for i, row in grouped.iterrows():
    ax.annotate(row['Year'], (row['Score_Std_Dev'], row['Top_10_Count']), textcoords="offset points", xytext=(0,5), ha='center')

# Calculate trend line
z = np.polyfit(grouped['Score_Std_Dev'], grouped['Top_10_Count'], 1)
p = np.poly1d(z)
plt.plot(grouped['Score_Std_Dev'], p(grouped['Score_Std_Dev']), "r--")

# Add trend line equation to the plot
plt.text(0.1, max(grouped['Top_10_Count']) - 1, f'y={z[0]:.2f}x+{z[1]:.2f}', color='red')

ax.grid(True)
plt.show()