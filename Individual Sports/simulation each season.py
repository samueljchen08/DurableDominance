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

def calculate_similarity(results, target_standings):
    similarity = 0
    for team, (wins, losses) in enumerate(target_standings):
        similarity += abs(results[team]['wins'] - wins) + abs(results[team]['losses'] - losses)
    return similarity

# Parameters
t = 23  # Number of teams
m = 82  # Number of matches each team plays
c = 2   # Control parameter
target_standings_new = [
    (63, 19), (58, 24), (55, 27), (57, 25), (52, 30), (46, 36), 
    (48, 34), (43, 39), (44, 38), (42, 40), (45, 37), (42, 40), 
    (46, 36), (46, 36), (39, 43), (35, 47), (34, 48), (33, 49), 
    (30, 52), (28, 54), (25, 57), (17, 65), (15, 67)
]

best_s_new = None
best_similarity_new = float('inf')
best_skills_new = None
best_results_new = None
best_match_results_new = None

for s in range(101):
    skills, results, match_results = simulate_league(t, m, s, c)
    similarity = calculate_similarity(results, target_standings_new)
    if similarity < best_similarity_new:
        best_similarity_new = similarity
        best_s_new = s
        best_skills_new = skills
        best_results_new = results
        best_match_results_new = match_results

best_s_new, best_similarity_new, best_skills_new, best_results_new, best_match_results_new