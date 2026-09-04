import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

#Open files
rankings = pd.read_excel("NFL.xlsx")

#Data manipulation
rankings.columns = ["Team", "Wins", "Losses", "Ties", "Win%", "Points", "Points_against", "MOV", "Year"]



#Calculate top 4 appearances
def calculate_historical_top4_appearances_with_unique_counts(df, current_season):
    historical_data = df[df['Year'] < current_season]
    if historical_data.empty:
        return pd.Series(dtype=float)

    top4_teams = historical_data[historical_data['Wins'] >= 12]
    unique_top4_teams_per_season = top4_teams.groupby('Year')['Team'].nunique()
    total_unique_top4_teams = unique_top4_teams_per_season.sum()

    top4_counts = top4_teams.groupby('Team').size()
    top4_percentages = (top4_counts / total_unique_top4_teams) * 100 if total_unique_top4_teams > 0 else pd.Series(dtype=float)

    return top4_percentages

print(calculate_historical_top4_appearances_with_unique_counts(rankings, 2000))

def calculate_dominance_index_with_breakdown(df):
    """
    Input a cleaned data file
    Uses helper function to find data points for all seasons and their historical finishes
    Return a tuple (dominance score of year (dictionary), Dictionary of years with values that are dictionaries of the top 4 teams dominance score)
    (1981:75, 1981: {Boston: 35, LA: 30, New York: 10})
    """
    seasons = df['Year'].unique()
    dominance_index = {}
    breakdown = {}

    for i in range(1, len(seasons)):
        current_season = seasons[i]
        historical_top4 = calculate_historical_top4_appearances_with_unique_counts(df, current_season)

        # Determine the top 4 teams for the current season
        top4_teams = df[df['Year'] == current_season].nlargest(4, 'Wins')['Team']

        # Ensure all top 4 teams have a historical percentage, default to 0 if missing
        top4_teams_historical = historical_top4.reindex(top4_teams, fill_value=0)

        # Calculate the dominance index for the current season
        index_value = top4_teams_historical.sum()
        dominance_index[current_season] = round(index_value,3)

        # Store the breakdown
        breakdown[current_season] = top4_teams_historical.to_dict()

    return dominance_index, breakdown



#Split apply combine algorithm
def competitiveness_index_win_pct(rankings):
    """
    Input: Cleaned data file
    Calculate standard deviation of teams win% in a given year rounded to 3 digit places
    Returns dictionary of years mapped to win% standard deviation
    """
    obj = rankings[rankings["Wins"]>0].groupby("Year")
    competitive_balance_dict = dict()
    for year in obj:
        var_for_year = 0
        win_list = list(year[1]["Win%"])
        for team_win_pct in win_list:
            var_for_year +=abs(team_win_pct - 0.5)
        sd_for_year = math.sqrt(var_for_year)
        competitive_balance_dict[year[0]] = round(sd_for_year,3)
    return competitive_balance_dict

dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
competitiveness_dict = competitiveness_index_win_pct(rankings)
dom_list = list(dominance_tuple[0].values())
comp_list = list(competitiveness_dict.values())
print(dom_list)
print(comp_list)


def all_nfl_dd():
    """
    1975 - Present Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_win_pct(rankings)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[:48],dom_list[:48])
    plt.xlabel('Standard Deviation of Win%')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[:48], dom_list[:48], 1)
    p = np.poly1d(z)
    plt.plot(comp_list[:48],p(comp_list[:48]),"r--")
    plt.title("1975 - Present NFL Win%")
    plt.show()

def nine_nfl_dd():
    """
    1990 - Present Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_win_pct(rankings)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[:33],dom_list[:33])
    plt.xlabel('Standard Deviation of Win%')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[:33], dom_list[:33], 1)
    p = np.poly1d(z)
    plt.plot(comp_list[:33],p(comp_list[:33]),"r--")
    plt.title("1990 - Present NFL Win%")
    plt.show()

def two_nfl_dd():
    """
    2000 - Present Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_win_pct(rankings)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[:23],dom_list[:23])
    plt.xlabel('Standard Deviation of Win%')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[:23], dom_list[:23], 1)
    p = np.poly1d(z)
    plt.plot(comp_list[:23],p(comp_list[:23]),"r--")
    plt.title("2000 - Present NFL Win%")
    plt.show()

# all_nfl_dd()
# nine_nfl_dd()
# two_nfl_dd()

def competitiveness_index_mov(rankings):
    obj = rankings[rankings["Wins"]>0].groupby("Year")
    competitive_balance_dict = dict()
    count = 0
    for year in obj:
        while count<3:
            count+=1
            continue
        var_net_rtg = 0
        net_rtg_list = list(year[1]["MOV"])
        for rating in net_rtg_list[3:]:
            var_net_rtg +=abs(rating)
        individual_var = var_net_rtg/len(rankings["Team"])
        sd_net_rtg = math.sqrt(individual_var)
        competitive_balance_dict[year[0]] = round(sd_net_rtg,3)
    return competitive_balance_dict


def all_nfl_dd_mov():
    """
    1975 - Present Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_mov(rankings)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[:48],dom_list[:48])
    plt.xlabel('Standard Deviation of MOV')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[:48], dom_list[:48], 1)
    p = np.poly1d(z)
    plt.plot(comp_list[:48],p(comp_list[:48]),"r--")
    plt.title("1975 - Present NFL MOV SD")
    plt.show()

def nine_nfl_dd_mov():
    """
    1990 - Present Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_mov(rankings)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[:33],dom_list[:33])
    plt.xlabel('Standard Deviation of MOV')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[:33], dom_list[:33], 1)
    p = np.poly1d(z)
    plt.plot(comp_list[:33],p(comp_list[:33]),"r--")
    plt.title("1990 - Present NFL MOV SD")
    plt.show()

def two_nfl_dd_mov():
    """
    2000 - Present Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_mov(rankings)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[:23],dom_list[:23])
    plt.xlabel('Standard Deviation of MOV')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[:23], dom_list[:23], 1)
    p = np.poly1d(z)
    plt.plot(comp_list[:23],p(comp_list[:23]),"r--")
    plt.title("2000 - Present NFL MOV SD")
    plt.show()

all_nfl_dd_mov()
nine_nfl_dd_mov()
two_nfl_dd_mov()