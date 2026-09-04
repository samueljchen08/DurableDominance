import numpy as np

 

def generate_skills(t, s, mean=50, lower_bound=0, upper_bound=100):

    skills = []

    for _ in range(t):

        skill = -1  # Initialize with an invalid skill

        while skill <= lower_bound or skill >= upper_bound:

            skill = np.random.normal(mean, s)

        skills.append(skill)

    return skills

 

def simulate_match(s1, s2, c):

    p1 = (s1 ** c) / (s1 ** c + s2 ** c)s

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

c = 2   # Control parameter

 

# Simulate the league

skills, results, match_results = simulate_league(t, m, s, c)

 

# Display the results of each match

print("Match Results:")

for match in match_results:

    team1, team2, winner = match

    print(f"Team {team1+1} vs Team {team2+1}: Winner is Team {winner+1}")

 

# Display the summary for each team

print("\nTeam Summary:")

for team in range(t):

    print(f"Team {team+1}: Skill = {skills[team]:.2f}, Wins = {results[team]['wins']}, Losses = {results[team]['losses']}")