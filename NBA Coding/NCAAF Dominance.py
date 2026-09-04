import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
file_path = 'NCAAF.xlsx'
data = pd.read_excel(file_path, header=2)

# Clean the data by removing rows where 'Year' is NaN
data_clean = data.dropna(subset=['Year'])

# Sort data by Year
data_clean = data_clean.sort_values(by='Year')

# Calculate measure of dominance
# Initialize a dictionary to keep track of past top 10 AP ranks
past_top_10_counts = {}

# Initialize a list to store dominance values for each year
dominance_list = []

for year in data_clean['Year'].unique():
    current_year_data = data_clean[data_clean['Year'] == year]
    top_10_teams = current_year_data[current_year_data['AP Rank'] <= 10]['School']
    
    # Count how many times these teams have been in the top 10 in the past
    current_dominance_count = sum(past_top_10_counts.get(team, 0) for team in top_10_teams)
    
    # Append the current year's dominance count to the list
    dominance_list.append(current_dominance_count)
    
    # Update the past top 10 counts for these teams
    for team in top_10_teams:
        if team in past_top_10_counts:
            past_top_10_counts[team] += 1
        else:
            past_top_10_counts[team] = 1

# Create a DataFrame for dominance values
dominance_df = pd.DataFrame({
    'Year': data_clean['Year'].unique(),
    'Dominance': dominance_list
})

# Calculate the measure of competitiveness: standard deviation of Win% for each year
competitiveness = data_clean.groupby('Year')['SRS'].std().reset_index()
competitiveness.columns = ['Year', 'Competitiveness']

# Combine both measures into a single DataFrame
measures = pd.merge(dominance_df, competitiveness, on='Year')

# Plot the data
plt.figure(figsize=(12, 6))
plt.scatter(measures['Dominance'], measures['Competitiveness'], alpha=0.7, label='Data points')

# Add a trend line
z = np.polyfit(measures['Dominance'], measures['Competitiveness'], 1)
p = np.poly1d(z)
plt.plot(measures['Dominance'], p(measures['Dominance']), "r--", label='Trend line')

plt.title('NCAA Football: Dominance vs Competitiveness by Year')
plt.xlabel('Measure of Dominance (Past top 10 AP rankings of current top 10 teams)')
plt.ylabel('Measure of Competitiveness (Std Dev of SRS)')
plt.legend()
plt.grid(True)
plt.show()