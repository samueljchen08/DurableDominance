import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

#Open files
matches = pd.read_excel("Combined_Team_Matches.xlsx")
rankings = pd.read_excel("Combined_Team_Rankings.xlsx")

#Data manipulation
matches.columns = ["Rk", "Team", "Season", "Opponent", "Result", "Tournament Winner"]
rankings.columns = ["Rank", "Team", "Conference", "Division", "Wins", "Losses", "Win%", "MOV", "ORTG", "DRTG", "NRTG", "Adjusted_MOV", "Adjusted_ORTG", "Adjusted_DRTG", "Adjusted_NRTG", "Year", "Misc"]



#Calculate top 4 appearances
def calculate_historical_top4_appearances_with_unique_counts(df, current_season):
    """
    Find the top 4 finishers in a given season and the historical top 4 finishes for them
    """
    df['Start_Year'] = df['Year'].str.split('-').str[0].astype(int)
    current_start_year = int(current_season.split('-')[0])
    
    # Filter historical data
    historical_data = df[df['Start_Year'] < current_start_year]
    if historical_data.empty:
        return pd.Series(dtype=float)

    top4_teams = historical_data[historical_data['Rank'] <= 4]
    unique_top4_teams_per_season = top4_teams.groupby('Year')['Team'].nunique()
    total_unique_top4_teams = unique_top4_teams_per_season.sum()

    top4_counts = top4_teams.groupby('Team').size()
    top4_percentages = (top4_counts / total_unique_top4_teams) * 100 if total_unique_top4_teams > 0 else pd.Series(dtype=float)

    return top4_percentages

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
        top4_teams = df[df['Year'] == current_season].nsmallest(4, 'Rank')['Team']

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





def all_nba_dd():
    """
    1981 - Present Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_win_pct(rankings)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[1:],dom_list)
    plt.xlabel('Standard Deviation of Win%')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[1:], dom_list, 1)
    p = np.poly1d(z)
    plt.plot(comp_list[1:],p(comp_list[1:]),"r--")
    plt.title("1981 - Present NBA Win%")
    plt.show()

def modern_nba_dd():
    """
    2000 - Present Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_win_pct(rankings)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[19:],dom_list[18:])
    plt.xlabel('Standard Deviation of Win%')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[19:], dom_list[18:], 1)
    p = np.poly1d(z)
    plt.plot(comp_list[19:],p(comp_list[19:]),"r--")
    plt.title("2000 - Present NBA Win%")
    plt.show()

def old_nba_dd():
    """
    1981-2000 Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_win_pct(rankings)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[1:19],dom_list[0:18])
    plt.xlabel('Standard Deviation of Win%')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[1:19], dom_list[0:18], 1)
    p = np.poly1d(z)
    plt.plot(comp_list[1:19],p(comp_list[1:19]),"r--")
    plt.title("1981 - 2000 NBA Win%")
    plt.show()

def ultra_modern_nba_dd():
    """
    2003 - Present Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_win_pct(rankings)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[22:],dom_list[21:])
    plt.xlabel('Standard Deviation of Win%')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[22:], dom_list[21:], 1)
    p = np.poly1d(z)
    plt.plot(comp_list[22:],p(comp_list[22:]),"r--")
    plt.title("2003 - Present NBA Win%")
    plt.show()


###############################


#Split apply combine algorithm
def competitiveness_index_net_rtg(rankings):
    obj = rankings[rankings["Wins"]>0].groupby("Year")
    competitive_balance_dict = dict()
    count = 0
    for year in obj:
        while count<3:
            count+=1
            continue
        var_net_rtg = 0
        net_rtg_list = list(year[1]["Adjusted_MOV"])
        for rating in net_rtg_list[3:]:
            var_net_rtg +=abs(rating)
        individual_var = var_net_rtg/len(rankings["Team"])
        sd_net_rtg = math.sqrt(individual_var)
        competitive_balance_dict[year[0]] = round(sd_net_rtg,3)
    return competitive_balance_dict

def eighties_nba_mov():
    """
    1988 - Present Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_net_rtg(rankings)
    print(len(dominance_tuple[0]))
    print(len(competitiveness_dict))
    print(competitiveness_dict)
    print(dominance_tuple)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[8:],dom_list[7:])
    plt.xlabel('Standard Deviation of MOV')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[8:], dom_list[7:], 1)
    p = np.poly1d(z)
    plt.plot(comp_list[8:],p(comp_list[8:]),"r--")
    plt.title("1988 - Present NBA MOV")
    plt.show()

def all_nba_mov():
    """
    1984 - Present Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_net_rtg(rankings)
    print(len(dominance_tuple[0]))
    print(len(competitiveness_dict))
    print(competitiveness_dict)
    print(dominance_tuple)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[4:],dom_list[3:])
    plt.xlabel('Standard Deviation of MOV')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[4:], dom_list[3:], 1)
    p = np.poly1d(z)
    plt.plot(comp_list[4:],p(comp_list[4:]),"r--")
    plt.title("1981 - Present NBA MOV")
    plt.show()

def two_thousands_mov():
    """
    2000 - Present Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_net_rtg(rankings)
    print(len(dominance_tuple[0]))
    print(len(competitiveness_dict))
    print(competitiveness_dict)
    print(dominance_tuple)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[20:],dom_list[19:])
    plt.xlabel('Standard Deviation of MOV')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[20:], dom_list[19:], 1)
    p = np.poly1d(z)
    plt.plot(comp_list[20:],p(comp_list[20:]),"r--")
    plt.title("2000 - Present Graph NBA MOV")
    plt.show()

def twenty_tens_mov():
    """
    2010 - Present Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_net_rtg(rankings)
    print(len(dominance_tuple[0]))
    print(len(competitiveness_dict))
    print(competitiveness_dict)
    print(dominance_tuple)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[30:],dom_list[29:])
    plt.xlabel('Standard Deviation of MOV')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[30:], dom_list[29:], 1)
    p = np.poly1d(z)
    plt.plot(comp_list[30:],p(comp_list[30:]),"r--")
    plt.title("2010 - Present Graph NBA MOV")
    plt.show()


########################################################


def competitiveness_index_net_rtg(rankings):
    obj = rankings[rankings["Wins"]>0].groupby("Year")
    competitive_balance_dict = dict()
    count = 0
    for year in obj:
        while count<3:
            count+=1
            continue
        var_net_rtg = 0
        net_rtg_list = list(year[1]["Adjusted_NRTG"])
        for rating in net_rtg_list[3:]:
            var_net_rtg +=abs(rating)
        individual_var = var_net_rtg/len(rankings["Team"])
        sd_net_rtg = math.sqrt(individual_var)
        competitive_balance_dict[year[0]] = round(sd_net_rtg,3)
    return competitive_balance_dict

def eighties_nba_net_rtg():
    """
    1988 - Present Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_net_rtg(rankings)
    print(len(dominance_tuple[0]))
    print(len(competitiveness_dict))
    print(competitiveness_dict)
    print(dominance_tuple)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[8:],dom_list[7:])
    plt.xlabel('Standard Deviation of NRTG')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[8:], dom_list[7:], 1)
    p = np.poly1d(z)
    plt.plot(comp_list[8:],p(comp_list[8:]),"r--")
    plt.title("1988 - Present NBA NRTG")
    plt.show()

def all_nba_net_rtg():
    """
    1984 - Present Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_net_rtg(rankings)
    print(len(dominance_tuple[0]))
    print(len(competitiveness_dict))
    print(competitiveness_dict)
    print(dominance_tuple)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[4:],dom_list[3:])
    plt.xlabel('Standard Deviation of NRTG')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[4:], dom_list[3:], 1)
    p = np.poly1d(z)
    plt.plot(comp_list[4:],p(comp_list[4:]),"r--")
    plt.title("1981 - Present NBA NRTG")
    plt.show()

def two_thousands_nba_net_rtg():
    """
    2000 - Present Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_net_rtg(rankings)
    print(len(dominance_tuple[0]))
    print(len(competitiveness_dict))
    print(competitiveness_dict)
    print(dominance_tuple)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[20:],dom_list[19:])
    plt.xlabel('Standard Deviation of NRTG')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[20:], dom_list[19:], 1)
    p = np.poly1d(z)
    plt.plot(comp_list[20:],p(comp_list[20:]),"r--")
    plt.title("2000 - Present Graph NBA NRTG")
    plt.show()

def twenty_tens_nba_net_rtg():
    """
    2010 - Present Graph
    """
    dominance_tuple = calculate_dominance_index_with_breakdown(rankings)
    competitiveness_dict = competitiveness_index_net_rtg(rankings)
    print(len(dominance_tuple[0]))
    print(len(competitiveness_dict))
    print(competitiveness_dict)
    print(dominance_tuple)
    dom_list = list(dominance_tuple[0].values())
    comp_list = list(competitiveness_dict.values())
    plt.scatter(comp_list[30:],dom_list[29:])
    plt.xlabel('Standard Deviation of NRTG')
    plt.ylabel('Dominance Index')
    z = np.polyfit(comp_list[30:], dom_list[29:], 1)
    p = np.poly1d(z)
    plt.plot(comp_list[30:],p(comp_list[30:]),"r--")
    plt.title("2010 - Present Graph NBA NRTG")
    plt.show()



def main():
    #Win% SD
    all_nba_dd()
    modern_nba_dd()
    ultra_modern_nba_dd()
    old_nba_dd()

    #Net Rating SD
    all_nba_net_rtg()
    eighties_nba_net_rtg()
    two_thousands_nba_net_rtg()
    twenty_tens_nba_net_rtg()

    #MOV SD
    all_nba_mov()
    eighties_nba_mov()
    two_thousands_mov()
    twenty_tens_mov()

main()