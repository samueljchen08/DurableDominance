import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'Mens_Tennis_Grand_Slam_Winner.csv'
data = pd.read_csv(file_path)

# Display the first few rows to verify
print(data.head())

# Assuming the data has columns for YEAR, Tournament, Winner Rank, and Historical Grand Slam Wins
# Calculate the average rank of the winners of the 4 grand slams each YEAR
average_rank_per_YEAR = data.groupby('YEAR')['Winner Rank'].mean().reset_index()

# Calculate the average number of historical grand slam wins for each winner each YEAR
average_historical_wins_per_YEAR = data.groupby('YEAR')['Historical Grand Slam Wins'].mean().reset_index()

# Merge the two dataframes on YEAR
merged_data = pd.merge(average_rank_per_YEAR, average_historical_wins_per_YEAR, on='YEAR')
merged_data.columns = ['YEAR', 'Average_Winner_Rank', 'Average_Historical_Wins']
a
# Scatter plot with average winner rank on x-axis and average historical grand slam wins on y-axis
plt.figure(figsize=(10, 6))
plt.scatter(merged_data['Average_Winner_Rank'], merged_data['Average_Historical_Wins'])

# Add labels and title
plt.xlabel('Average Rank of Winners')
plt.ylabel('Average Historical Grand Slam Wins')
plt.title('Average Rank of Winners vs Average Historical Grand Slam Wins by YEAR')

# Annotate each point with the corresponding YEAR
for i, row in merged_data.iterrows():
    plt.annotate(row['YEAR'], (row['Average_Winner_Rank'], row['Average_Historical_Wins']), textcoords="offset points", xytext=(0,5), ha='center')

plt.grid(True)
plt.show()