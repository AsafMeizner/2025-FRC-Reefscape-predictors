import json
import random
import time
import os
import math
import pandas as pd

#########################################
# Part 1: Data Generation
#########################################
def generate_data(filename="scouting_data_intelligent.json"):
    """
    Generate synthetic match data for Reefscape.
    Modifications:
      - Each alliance can score up to 12 corals per level.
      - Algae totals can be up to 36.
      - No initial blocking is applied on coral levels.
    """
    def random_partition(total, parts):
        if parts == 1:
            return [total]
        cuts = sorted([random.randint(0, total) for _ in range(parts - 1)])
        values = []
        previous = 0
        for cut in cuts:
            values.append(cut - previous)
            previous = cut
        values.append(total - previous)
        return values

    NUM_TEAMS = 30
    MATCHES_PER_TEAM = 10
    total_team_entries = NUM_TEAMS * MATCHES_PER_TEAM
    remainder = total_team_entries % 6
    if remainder != 0:
        total_team_entries += (6 - remainder)
    num_matches = total_team_entries // 6

    team_assignments = [team for team in range(1, NUM_TEAMS + 1) for _ in range(MATCHES_PER_TEAM)]
    while len(team_assignments) < total_team_entries:
        team_assignments.append(random.randint(1, NUM_TEAMS))
    random.shuffle(team_assignments)

    current_time = int(time.time() * 1000)
    submissions = []
    global_team_index = 0

    for match in range(1, num_matches + 1):
        alliance_data = {}
        for alliance in ['A', 'B']:
            alliance_data[alliance] = {}
            for level in ['L2', 'L3', 'L4']:
                total_alliance = random.randint(0, 12)  # up to 12 corals per level
                aut_total = random.randint(0, total_alliance)
                teleop_total = total_alliance - aut_total
                aut_parts = random_partition(aut_total, 3)
                teleop_parts = random_partition(teleop_total, 3)
                alliance_data[alliance][level] = {'aut': aut_parts, 'teleop': teleop_parts}
        for alliance_index, alliance in enumerate(['A', 'B']):
            for robot_index in range(3):
                submission = {}
                submission["scouter"] = f"Scouter {robot_index + alliance_index * 3 + 1}"
                submission["matchNumber"] = match
                submission["teamNumber"] = team_assignments[global_team_index]
                global_team_index += 1
                submission["noShow"] = random.random() < 0.05

                # Autonomous Section
                submission["Mved"] = random.choice([True, False])
                submission["L1sc"] = random.randint(0, 10)  # L1 unlimited in auto
                for level in ['L2', 'L3', 'L4']:
                    submission[f"{level}sc"] = alliance_data[alliance][level]['aut'][robot_index]
                submission["L1ms"] = random.randint(0, 2)
                for level in ['L2', 'L3', 'L4']:
                    submission[f"{level}ms"] = random.randint(0, 2)
                submission["InG"] = random.randint(0, 5)
                submission["InS"] = random.randint(0, 5)
                submission["RmAr"] = random.randint(0, 5)
                submission["RmAg"] = random.randint(0, 5)
                submission["ScAb"] = random.randint(0, 3)
                submission["ScAp"] = random.randint(0, 3)

                # TeleOp Section
                submission["tL1sc"] = random.randint(0, 15)  # L1 in teleop
                for level in ['L2', 'L3', 'L4']:
                    submission[f"t{level}sc"] = alliance_data[alliance][level]['teleop'][robot_index]
                submission["tL1ms"] = random.randint(0, 2)
                for level in ['L2', 'L3', 'L4']:
                    submission[f"t{level}ms"] = random.randint(0, 2)
                submission["tInG"] = random.randint(0, 10)
                submission["tInS"] = random.randint(0, 10)
                submission["tRmAr"] = random.randint(0, 10)
                submission["tRmAg"] = random.randint(0, 10)
                submission["tScAb"] = random.randint(0, 5)
                submission["tScAp"] = random.randint(0, 5)
                submission["Fou"] = random.randint(0, 2)

                # End Game Section
                submission["epo"] = random.choice(["No", "P", "Sc", "Dc", "Fd", "Fs"])

                # Post-Match
                submission["dto"] = random.random() < 0.1
                submission["yc"] = random.choices(["No Card", "Yellow", "Red"], weights=[90, 8, 2])[0]
                submission["co"] = random.choice(["", "Good performance", "Minor issues", 
                                                   "Robot was stuck", "Exceeded expectations", "Needs improvement"])
                submission["submissionTime"] = current_time + random.randint(0, 1000000)
                submissions.append(submission)

    with open(filename, "w") as f:
        json.dump(submissions, f, indent=2)
    print(f"Generated {len(submissions)} submissions across {num_matches} matches.")

#########################################
# Part 2: Calculation-Based Predictor Functions
#########################################
def average_auto_score(data, teamNumber):
    """Calculates average autonomous coral score using multipliers: L1=3, L2=4, L3=6, L4=7."""
    teamData = [entry for entry in data if entry["teamNumber"] == teamNumber]
    if not teamData:
        return 0
    total = 0
    for entry in teamData:
        total += (entry.get("L1sc",0)*3 + entry.get("L2sc",0)*4 +
                  entry.get("L3sc",0)*6 + entry.get("L4sc",0)*7)
    return total / len(teamData)

def average_teleop_score(data, teamNumber):
    """Calculates average teleop coral score using multipliers: L1=2, L2=3, L3=4, L4=5."""
    teamData = [entry for entry in data if entry["teamNumber"] == teamNumber]
    if not teamData:
        return 0
    total = 0
    for entry in teamData:
        total += (entry.get("tL1sc",0)*2 + entry.get("tL2sc",0)*3 +
                  entry.get("tL3sc",0)*4 + entry.get("tL4sc",0)*5)
    return total / len(teamData)

def average_algae_score(data, teamNumber):
    """
    Calculates average algae score combining auto and teleop.
    (Uses both barge and processor counts.)
    """
    teamData = [entry for entry in data if entry["teamNumber"] == teamNumber]
    if not teamData:
        return 0
    total = 0
    for entry in teamData:
        total += (entry.get("ScAb",0)*4 + entry.get("ScAp",0)*6 +
                  entry.get("tScAb",0)*4 + entry.get("tScAp",0)*6)
    return total / len(teamData)

def average_algae_barge(data, teamNumber):
    """Calculates average algae barge score (only barge, 4 points each)."""
    teamData = [entry for entry in data if entry["teamNumber"] == teamNumber]
    if not teamData:
        return 0
    total = 0
    for entry in teamData:
        total += (entry.get("ScAb",0)*4 + entry.get("tScAb",0)*4)
    return total / len(teamData)

def average_endgame_points(data, teamNumber):
    """Calculates average endgame points based on the 'epo' field."""
    mapping = {"P": 2, "Sc": 6, "Dc": 12, "No": 0, "Fs": 0, "Fd": 0}
    teamData = [entry for entry in data if entry["teamNumber"] == teamNumber]
    if not teamData:
        return 0
    total = sum(mapping.get(entry.get("epo", "No"), 0) for entry in teamData)
    return total / len(teamData)

def level_average(data, teamNumber, level):
    """Calculates average total corals (auto+teleop) scored on a given level for a team."""
    teamData = [entry for entry in data if entry["teamNumber"] == teamNumber]
    if not teamData:
        return 0
    total = 0
    for entry in teamData:
        total += entry.get(f"{level}sc",0) + entry.get(f"t{level}sc",0)
    return total / len(teamData)

def autonomous_bonus(data, teamNumber):
    """
    Returns 1 if the team in autonomous:
      - Leaves the starting line (Mved true in ≥50% of matches)
      - Scores at least one coral (any level)
    Otherwise, returns 0.
    """
    teamData = [entry for entry in data if entry["teamNumber"] == teamNumber]
    if not teamData:
        return 0
    mved_rate = sum(1 for entry in teamData if entry.get("Mved", False)) / len(teamData)
    auto_coral = sum(entry.get("L1sc",0) + entry.get("L2sc",0) + entry.get("L3sc",0) + entry.get("L4sc",0)
                     for entry in teamData)
    return 1 if mved_rate >= 0.5 and auto_coral > 0 else 0

def alliance_autonomous_bonus(data, team_numbers):
    """Returns 1 if every team in the alliance meets the autonomous bonus condition."""
    return 1 if all(autonomous_bonus(data, t) == 1 for t in team_numbers) else 0

def alliance_coral_bonus(data, team_numbers):
    """
    Returns 1 if for each coral level (L1-L4) the alliance (sum of averages)
    scores at least 5 corals; otherwise returns 0.
    """
    levels = ["L1", "L2", "L3", "L4"]
    for lvl in levels:
        total = sum(level_average(data, t, lvl) for t in team_numbers)
        if total < 5:
            return 0
    return 1

def alliance_barge_bonus(data, team_numbers):
    """
    Returns 1 if the alliance's average barge points (sum over teams) exceeds 14.
    """
    total = sum(average_algae_barge(data, t) for t in team_numbers)
    return 1 if total > 14 else 0

def calculate_estimated_alliance_score(data, team_numbers):
    """
    Calculation-based predictor for alliance score and RP.
    
    Estimated Total Points = Sum over teams of:
         (average_auto_score + average_teleop_score + average_algae_score + average_endgame_points)
    
    Estimated RP is computed as follows (max 6 RP):
         - Autonomous Bonus: 1 RP if every team meets the autonomous condition.
         - Coral Bonus: 1 RP if alliance scores at least 5 corals on each level.
         - Barge Bonus: 1 RP if alliance barge points > 14.
         - Win Bonus: Assume win bonus of 3 RP.
         Total RP = (autonomous + coral + barge + win), capped at 6.
    """
    total_points = 0
    for t in team_numbers:
        total_points += (average_auto_score(data, t) +
                         average_teleop_score(data, t) +
                         average_algae_score(data, t) +
                         average_endgame_points(data, t))
    
    rp_autonomous = alliance_autonomous_bonus(data, team_numbers)
    rp_coral = alliance_coral_bonus(data, team_numbers)
    rp_barge = alliance_barge_bonus(data, team_numbers)
    rp_win = 3  # Assuming the alliance wins
    rp_total = rp_autonomous + rp_coral + rp_barge + rp_win
    if rp_total > 6:
        rp_total = 6

    return total_points, rp_total

#########################################
# Part 3: Main Flow for Prediction
#########################################
def main():
    data_file = "scouting_data_intelligent.json"
    if not os.path.exists(data_file):
        print("Data file not found; generating synthetic data...")
        generate_data(data_file)
    else:
        print(f"Using existing data file: {data_file}")

    with open(data_file, "r") as f:
        data = json.load(f)

    # For demonstration, choose an alliance (for example, teams 20, 11, and 28).
    alliance = [20, 11, 28]
    predicted_points, predicted_rp = calculate_estimated_alliance_score(data, alliance)

    print("\n=== Calculation-Based Alliance Prediction ===")
    print("Alliance Teams:", alliance)
    print("Estimated Total Points:", round(predicted_points, 2))
    print("Estimated Ranking Points (RP):", predicted_rp)

if __name__ == '__main__':
    main()
