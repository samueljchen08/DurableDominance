import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'Combined_Team_Rankings.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')  # Replace 'Sheet1' with the actual sheet name if different

# Display the first few rows to verify
print(data.head())

# Get the list of years
years = data['Year_'].unique()

# Plot the number of wins for each team for each year
for year in years:
    year_data = data[data['Year_'] == year]
    teams = year_data['Unnamed: 1_level_0_Team']
    wins = year_data['Unnamed: 4_level_0_W']
    
    plt.figure(figsize=(10, 6))
    plt.bar(teams, wins)
    plt.xlabel('Team')
    plt.ylabel('Number of Wins')
    plt.title(f'Number of Wins for Each Team in {year}')
    plt.xticks(rotation=90)
    plt.show()
