import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
file_path = 'Tour De France.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')  # Replace 'Sheet1' with the actual sheet name if different

# Display the first few rows to verify
print(data.head())

# Convert the 'Times' column from string format to timedelta
data['Times'] = pd.to_timedelta(data['Times'])

# Calculate the standard deviation of times for each year
std_dev_time_per_year = data.groupby('Year')['Times'].std().reset_index()

# Calculate the sum of historical top 5 finishes for all riders each year
top_5 = data[data['Position'].isin([1, 2, 3, 4, 5])]
historical_top_5_sum_per_year = top_5.groupby('Year')['Historical Top 5 Finishes'].sum().reset_index()

# Merge the two dataframes on Year
merged_data = pd.merge(historical_top_5_sum_per_year, std_dev_time_per_year, on='Year')
merged_data.columns = ['Year', 'Historical_Top_5_Sum', 'Std_Dev_Times']

# Convert the timedelta to total seconds for plotting
merged_data['Std_Dev_Times'] = merged_data['Std_Dev_Times'].dt.total_seconds()

# Scatter plot with sum of historical top 5 finishes on x-axis and standard deviation of times on y-axis
plt.figure(figsize=(10, 6))
plt.scatter(merged_data['Historical_Top_5_Sum'], merged_data['Std_Dev_Times'])

# Add labels and title
plt.xlabel('Sum of Historical Top 5 Finishes')
plt.ylabel('Standard Deviation of Times (seconds)')
plt.title('Sum of Historical Top 5 Finishes vs Standard Deviation of Times by Year')

# Annotate each point with the corresponding year
for i, row in merged_data.iterrows():
    plt.annotate(row['Year'], (row['Historical_Top_5_Sum'], row['Std_Dev_Times']), textcoords="offset points", xytext=(0,5), ha='center')

plt.grid(True)
plt.show()