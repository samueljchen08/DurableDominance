import numpy as np
import matplotlib.pyplot as plt

def generate_skills(t, s, mean=50, lower_bound=0, upper_bound=100):
    skills = []
    for _ in range(t):
        skill = -1  # Initialize with an invalid skill
        while skill <= lower_bound or skill >= upper_bound:
            skill = np.random.normal(mean, s)
        skills.append(skill)
    return skills

def simulate_match(s1, s2, c):
    p1 = (s1 ** c) / (s1 ** c + s2 ** c)
    return np.random.rand() < p1

def create_schedule(t, m):
    teams = list(range(t))
    matches = []
    match_counts = {team: 0 for team in teams}
    while len(matches) < t * m // 2:
        for i in range(t):
            for j in range(i + 1, t):
                if match_counts[i] < m and match_counts[j] < m:
                    matches.append((i, j))
                    match_counts[i] += 1
                    match_counts[j] += 1
                    if len(matches) == t * m // 2:
                        break
            if len(matches) == t * m // 2:
                break
    return matches

def simulate_league(t, m, s, c):
    skills = generate_skills(t, s)
    schedule = create_schedule(t, m)
    results = {team: {'wins': 0, 'losses': 0} for team in range(t)}
    match_results = []

    for match in schedule:
        team1, team2 = match
        if simulate_match(skills[team1], skills[team2], c):
            results[team1]['wins'] += 1
            results[team2]['losses'] += 1
            match_results.append((team1, team2, team1))
        else:
            results[team2]['wins'] += 1
            results[team1]['losses'] += 1
            match_results.append((team1, team2, team2))

    return skills, results, match_results

# Parameters
t = 30  # Number of teams
m = 82  # Number of matches each team plays
s = 20  # Standard deviation of skill
c_values = np.arange(0, 2.2, 0.2)  # Control parameter values

# Run simulations and plot results
for c in c_values:
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    fig.suptitle(f'Simulation Results for c = {c}', fontsize=16)
    
    for i in range(10):
        skills, results, match_results = simulate_league(t, m, s, c)
        
        wins = [results[team]['wins'] for team in range(t)]
        losses = [results[team]['losses'] for team in range(t)]
        
        ax = axs[i // 5, i % 5]
        ax.bar(range(t), wins, label='Wins')
        ax.bar(range(t), losses, bottom=wins, label='Losses')
        ax.set_title(f'Simulation {i + 1}')
        ax.set_xlabel('Team')
        ax.set_ylabel('Number of Matches')
        if i == 0:
            ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()