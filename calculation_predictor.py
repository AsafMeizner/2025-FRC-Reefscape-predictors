import json
import random
import time
import os
import math
import numpy as np
import pandas as pd

#########################################
# Part 1: Data Generation
#########################################
def generate_data(filename="scouting_data_intelligent.json"):
    """
    Generate synthetic match data for Reefscape.
    • Each alliance can score up to 12 corals per level.
    • Algae totals can be up to 36.
    • No initial blocking is applied on coral levels.
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
    total = sum(entry.get("L1sc", 0) * 3 + entry.get("L2sc", 0) * 4 +
                entry.get("L3sc", 0) * 6 + entry.get("L4sc", 0) * 7 for entry in teamData)
    return total / len(teamData)

def average_teleop_score(data, teamNumber):
    """Calculates average teleop coral score using multipliers: L1=2, L2=3, L3=4, L4=5."""
    teamData = [entry for entry in data if entry["teamNumber"] == teamNumber]
    if not teamData:
        return 0
    total = sum(entry.get("tL1sc", 0) * 2 + entry.get("tL2sc", 0) * 3 +
                entry.get("tL3sc", 0) * 4 + entry.get("tL4sc", 0) * 5 for entry in teamData)
    return total / len(teamData)

def average_algae_score(data, teamNumber):
    """
    Calculates average algae score combining auto and teleop.
    (Uses both barge (4 pts) and processor (6 pts) counts.)
    """
    teamData = [entry for entry in data if entry["teamNumber"] == teamNumber]
    if not teamData:
        return 0
    total = sum(entry.get("ScAb", 0) * 4 + entry.get("ScAp", 0) * 6 +
                entry.get("tScAb", 0) * 4 + entry.get("tScAp", 0) * 6 for entry in teamData)
    return total / len(teamData)

def average_algae_barge(data, teamNumber):
    """Calculates average algae barge score (barge only, 4 points each)."""
    teamData = [entry for entry in data if entry["teamNumber"] == teamNumber]
    if not teamData:
        return 0
    total = sum(entry.get("ScAb", 0) * 4 + entry.get("tScAb", 0) * 4 for entry in teamData)
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
    total = sum(entry.get(f"{level}sc", 0) + entry.get(f"t{level}sc", 0) for entry in teamData)
    return total / len(teamData)

def level_max(data, teamNumber, level):
    """Calculates the maximum total corals (auto+teleop) scored on a given level for a team."""
    teamData = [entry for entry in data if entry["teamNumber"] == teamNumber]
    if not teamData:
        return 0
    return max(entry.get(f"{level}sc", 0) + entry.get(f"t{level}sc", 0) for entry in teamData)

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
    auto_coral = sum(entry.get("L1sc", 0) + entry.get("L2sc", 0) +
                     entry.get("L3sc", 0) + entry.get("L4sc", 0) for entry in teamData)
    return 1 if mved_rate >= 0.5 and auto_coral > 0 else 0

def alliance_autonomous_bonus(data, team_numbers):
    """Returns 1 if every team in the alliance meets the autonomous bonus condition."""
    return 1 if all(autonomous_bonus(data, t) == 1 for t in team_numbers) else 0

def alliance_coral_bonus(data, team_numbers, threshold=5):
    """
    Returns 1 if for each coral level (L1-L4) the alliance (sum of per-team averages)
    does not exceed the physical limit (12) but meets at least the given threshold.
    For each level, we sum each team's average corals on that level and then cap at 12.
    """
    levels = ["L1", "L2", "L3", "L4"]
    for lvl in levels:
        total = sum(level_average(data, t, lvl) for t in team_numbers)
        capped = min(total, 12)
        if capped < threshold:
            return 0
    return 1

def alliance_barge_bonus(data, team_numbers):
    """
    Returns 1 if the alliance's average barge points (sum over teams) exceeds 14.
    """
    total = sum(average_algae_barge(data, t) for t in team_numbers)
    return 1 if total > 14 else 0

#########################################
# Part 2a: Parameter Tuning for RP Mode
#########################################
def tune_rp_parameters(data, team_numbers, blend_range, multiplier_range):
    """
    Grid-search over blend_factor and avg_multiplier candidate values.
    Our objective is lexicographic:
      1. Maximize estimated RP (primary objective; ideally 6)
      2. Among candidates with equal RP, maximize estimated points.
    Returns the best blend_factor, best multiplier, best estimated points, and best estimated RP.
    """
    best_rp = -math.inf
    best_points = -math.inf
    best_blend = None
    best_mult = None
    for blend in blend_range:
        for mult in multiplier_range:
            est_points, est_rp, _ = calculate_estimated_alliance_score(data, team_numbers, mode="rp", blend_factor=blend, avg_multiplier=mult)
            # Primary objective: maximize RP (capped at 6)
            if est_rp > best_rp:
                best_rp = est_rp
                best_points = est_points
                best_blend = blend
                best_mult = mult
            # If RP tie, then maximize points
            elif est_rp == best_rp and est_points > best_points:
                best_points = est_points
                best_blend = blend
                best_mult = mult
    return best_blend, best_mult, best_points, best_rp

#########################################
# Part 3: Calculation-Based Predictor Functions (with tunable RP parameters)
#########################################
def calculate_estimated_alliance_score(data, team_numbers, mode="points", blend_factor=0.75, avg_multiplier=4.5):
    """
    Calculation-based predictor for alliance score and RP.
    
    Two modes:
      - mode="points": The alliance focuses on maximizing points.
         Estimated Total Points = Sum over teams of:
             (average_auto_score + average_teleop_score + average_algae_score + average_endgame_points)
         RP Conditions:
             - Coral Bonus: if each level (L1-L4) totals at least 5 corals (capped at 12).
             - Autonomous Bonus: if every team meets the autonomous condition.
             - Barge Bonus: if alliance barge points > 14.
             - Win Bonus: fixed 3 RP.
      
      - mode="rp": The alliance focuses on meeting RP thresholds.
         In this mode we assume the alliance first scores the minimum required for RP,
         then uses any extra potential. We compute:
             full_points = sum of team full average scores (as in points mode)
             min_required = coral_thresh * 4 * avg_multiplier * 2  (for 4 levels, 2 phases)
             estimated_points = min_required + blend_factor * max(0, full_points - min_required)
         Where coral_thresh = 3 if average processor algae score (from ScAp/tScAp) ≥ 12, else 5.
    
    Returns a tuple: (estimated_total_points, estimated_RP, report_dict)
    """
    report = {"teams": {}}
    
    for t in team_numbers:
        team_report = {}
        auto = average_auto_score(data, t)
        teleop = average_teleop_score(data, t)
        algae = average_algae_score(data, t)
        endgame = average_endgame_points(data, t)
        total = auto + teleop + algae + endgame
        team_report["Average Autonomous Coral Score"] = auto
        team_report["Average TeleOp Coral Score"] = teleop
        team_report["Average Algae Score"] = algae
        team_report["Average Endgame Points"] = endgame
        team_report["Total Average Score"] = total
        per_level = {lvl: level_average(data, t, lvl) for lvl in ["L1", "L2", "L3", "L4"]}
        team_report["Per-Level Averages"] = per_level
        team_report["Autonomous Bonus Condition"] = "Yes" if autonomous_bonus(data, t) == 1 else "No"
        report["teams"][t] = team_report

    alliance_total = sum(report["teams"][t]["Total Average Score"] for t in team_numbers)
    
    if mode == "points":
        estimated_points = alliance_total
        coral_thresh = 5
    else:  # mode == "rp"
        def average_processor(data, t):
            teamData = [entry for entry in data if entry["teamNumber"] == t]
            if not teamData:
                return 0
            return sum(entry.get("ScAp", 0) * 6 + entry.get("tScAp", 0) * 6 for entry in teamData) / len(teamData)
        processor_avg = sum(average_processor(data, t) for t in team_numbers) / len(team_numbers)
        coral_thresh = 3 if processor_avg >= 12 else 5
        full_points = alliance_total
        min_required = coral_thresh * 4 * avg_multiplier * 2  # 4 levels, 2 phases
        estimated_points = min_required + blend_factor * max(0, full_points - min_required)
    
    rp_coral = alliance_coral_bonus(data, team_numbers, threshold=coral_thresh)
    rp_auto = alliance_autonomous_bonus(data, team_numbers)
    rp_barge = alliance_barge_bonus(data, team_numbers)
    rp_win = 3
    rp_total = rp_coral + rp_auto + rp_barge + rp_win
    if rp_total > 6:
        rp_total = 6

    report["Alliance Totals"] = {"Combined Estimated Total Points": estimated_points}
    report["RP Conditions"] = {
        "Autonomous Bonus": rp_auto,
        "Coral Bonus (threshold=" + str(coral_thresh) + ")": rp_coral,
        "Barge Bonus": rp_barge,
        "Win Bonus": rp_win,
        "Total Estimated RP (capped at 6)": rp_total
    }
    report["Overall Estimated Total Points"] = estimated_points
    report["Overall Estimated RP"] = rp_total
    report["Mode"] = mode
    report["Processor Average (for RP mode decision)"] = processor_avg if mode == "rp" else None

    return estimated_points, rp_total, report

#########################################
# Part 3a: Baseline Physical Performance Calculation
#########################################
def baseline_alliance_physical(data, team_numbers):
    """
    Computes the alliance's baseline average and maximum scores subject to physical limitations.
    For each coral level (L1-L4), sum each team's average and maximum scores, capping the sum at 12.
    Apply approximate multipliers (L1:2.5, L2:3.5, L3:5.0, L4:6.0),
    then add the alliance's algae and endgame points.
    Returns: (baseline_average, baseline_max, difference).
    """
    multipliers = {"L1": 2.5, "L2": 3.5, "L3": 5.0, "L4": 6.0}
    
    alliance_avg_coral = 0
    alliance_max_coral = 0
    for lvl in ["L1", "L2", "L3", "L4"]:
        total_avg_lvl = sum(level_average(data, t, lvl) for t in team_numbers)
        total_max_lvl = sum(level_max(data, t, lvl) for t in team_numbers)
        capped_avg = min(total_avg_lvl, 12)
        capped_max = min(total_max_lvl, 12)
        alliance_avg_coral += capped_avg * multipliers[lvl]
        alliance_max_coral += capped_max * multipliers[lvl]
    
    alliance_avg_algae = sum(average_algae_score(data, t) for t in team_numbers)
    alliance_avg_endgame = sum(average_endgame_points(data, t) for t in team_numbers)
    
    baseline_avg = alliance_avg_coral + alliance_avg_algae + alliance_avg_endgame
    
    def team_max_algae(data, t):
        teamData = [entry for entry in data if entry["teamNumber"] == t]
        if not teamData:
            return 0
        return max(entry.get("ScAb", 0) * 4 + entry.get("ScAp", 0) * 6 +
                   entry.get("tScAb", 0) * 4 + entry.get("tScAp", 0) * 6 for entry in teamData)
    def team_max_endgame(data, t):
        mapping = {"P": 2, "Sc": 6, "Dc": 12, "No": 0, "Fs": 0, "Fd": 0}
        teamData = [entry for entry in data if entry["teamNumber"] == t]
        if not teamData:
            return 0
        return max(mapping.get(entry.get("epo", "No"), 0) for entry in teamData)
    
    alliance_max_algae = sum(team_max_algae(data, t) for t in team_numbers)
    alliance_max_endgame = sum(team_max_endgame(data, t) for t in team_numbers)
    
    baseline_max = alliance_max_coral + alliance_max_algae + alliance_max_endgame
    diff = baseline_max - baseline_avg
    return baseline_avg, baseline_max, diff

#########################################
# Part 4: Simulation of Multiple Alliances
#########################################
def simulate_alliances(data, team_stats_pool, num_alliances=5):
    all_teams = list(team_stats_pool.keys())
    results = []
    for i in range(num_alliances):
        alliance = random.sample(all_teams, 3)
        baseline_avg, baseline_max, diff = baseline_alliance_physical(data, alliance)
        pts_mode_points, pts_mode_rp, _ = calculate_estimated_alliance_score(data, alliance, mode="points")
        rp_mode_points, rp_mode_rp, _ = calculate_estimated_alliance_score(data, alliance, mode="rp")
        alliance_result = {
            "Alliance Teams": alliance,
            "Baseline Average Score": round(baseline_avg, 2),
            "Baseline Maximum Score": round(baseline_max, 2),
            "Baseline Difference (Max - Average)": round(diff, 2),
            "Calculated (Points Mode) - Estimated Points": round(pts_mode_points, 2),
            "Calculated (Points Mode) - Estimated RP": pts_mode_rp,
            "Calculated (RP Mode) - Estimated Points": round(rp_mode_points, 2),
            "Calculated (RP Mode) - Estimated RP": rp_mode_rp
        }
        results.append(alliance_result)
    return results

#########################################
# Part 5: Parameter Tuning and Main Flow
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
    
    # Detailed prediction for a chosen alliance
    alliance = [20, 11, 28]
    print("\n=== Calculation-Based Alliance Detailed Prediction Report (Points Mode) ===")
    pts_points, pts_rp, report_points = calculate_estimated_alliance_score(data, alliance, mode="points")
    for key, value in report_points.items():
        if key == "teams":
            print("Detailed Per-Team Averages:")
            for team, rep in value.items():
                print(f"Team {team}:")
                for k, v in rep.items():
                    if isinstance(v, dict):
                        print("  " + k + ":")
                        for lvl, avg in v.items():
                            print(f"    {lvl}: {avg:.2f}")
                    else:
                        print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

    print("\n=== Calculation-Based Alliance Detailed Prediction Report (RP Mode) ===")
    # Tuning RP mode parameters: we use a grid search over blend factor and avg multiplier.
    blend_range = np.linspace(0.5, 1.5, 21)  # from 0.5 to 1.5
    multiplier_range = np.linspace(3.0, 6.0, 21)  # from 3.0 to 6.0
    best_blend, best_mult, best_points, best_rp = tune_rp_parameters(data, alliance, blend_range, multiplier_range)
    print(f"Tuned RP Mode Parameters for alliance {alliance}: Best Blend Factor = {best_blend}, Best Avg Multiplier = {best_mult} (Estimated Points = {round(best_points,2)}, RP = {best_rp})")
    rp_points, rp_rp, report_rp = calculate_estimated_alliance_score(data, alliance, mode="rp", blend_factor=best_blend, avg_multiplier=best_mult)
    for key, value in report_rp.items():
        if key == "teams":
            print("Detailed Per-Team Averages:")
            for team, rep in value.items():
                print(f"Team {team}:")
                for k, v in rep.items():
                    if isinstance(v, dict):
                        print("  " + k + ":")
                        for lvl, avg in v.items():
                            print(f"    {lvl}: {avg:.2f}")
                    else:
                        print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

    baseline_avg, baseline_max, diff = baseline_alliance_physical(data, alliance)
    print("\n=== Baseline Alliance Performance (For Alliance", alliance, ") ===")
    print("Baseline Average Score (capped by physical limits):", round(baseline_avg, 2))
    print("Baseline Maximum Score (capped by physical limits):", round(baseline_max, 2))
    print("Difference (Max - Average):", round(diff, 2))
    
    # Build team_stats_pool from aggregated data for simulation
    df = pd.DataFrame(data)
    agg_df = df.groupby("teamNumber").agg({
        'L1sc':'sum','L1ms':'sum',
        'L2sc':'sum','L2ms':'sum',
        'L3sc':'sum','L3ms':'sum',
        'L4sc':'sum','L4ms':'sum',
        'tL1sc':'sum','tL1ms':'sum',
        'tL2sc':'sum','tL2ms':'sum',
        'tL3sc':'sum','tL3ms':'sum',
        'tL4sc':'sum','tL4ms':'sum',
        'ScAb':'mean','ScAp':'mean',
        'tScAb':'mean','tScAp':'mean',
        'Mved':'mean'
    })
    def compute_rate(sc, ms):
        total = sc + ms
        return sc/total if total > 0 else 0.5
    team_stats_pool = {}
    for team, row in agg_df.iterrows():
        team_stats_pool[team] = {
            'auto_coral_L1_rate': compute_rate(row['L1sc'], row['L1ms']),
            'auto_coral_L1_cycle': 8.0,
            'auto_coral_L2_rate': compute_rate(row['L2sc'], row['L2ms']),
            'auto_coral_L2_cycle': 10.0,
            'auto_coral_L3_rate': compute_rate(row['L3sc'], row['L3ms']),
            'auto_coral_L3_cycle': 12.0,
            'auto_coral_L4_rate': compute_rate(row['L4sc'], row['L4ms']),
            'auto_coral_L4_cycle': 15.0,
            'teleop_coral_L1_rate': compute_rate(row['tL1sc'], row['tL1ms']),
            'teleop_coral_L1_cycle': 6.0,
            'teleop_coral_L2_rate': compute_rate(row['tL2sc'], row['tL2ms']),
            'teleop_coral_L2_cycle': 8.0,
            'teleop_coral_L3_rate': compute_rate(row['tL3sc'], row['tL3ms']),
            'teleop_coral_L3_cycle': 10.0,
            'teleop_coral_L4_rate': compute_rate(row['tL4sc'], row['tL4ms']),
            'teleop_coral_L4_cycle': 12.0,
            'auto_algae_barge_rate': 0.8, 'auto_algae_barge_cycle': 8.0,
            'auto_algae_processor_rate': 0.7, 'auto_algae_processor_cycle': 8.0,
            'teleop_algae_barge_rate': 0.85, 'teleop_algae_barge_cycle': 6.0,
            'teleop_algae_processor_rate': 0.75, 'teleop_algae_processor_cycle': 6.0,
            'auto_left': 1.0 if row['Mved'] >= 0.5 else 0.0
        }
    
    print("\n=== Simulation of Multiple Alliances ===")
    sim_results = simulate_alliances(data, team_stats_pool, num_alliances=5)
    for res in sim_results:
        print("--------------------------------------------------")
        print("Alliance Teams:", res["Alliance Teams"])
        print("Baseline Average Score:", res["Baseline Average Score"])
        print("Baseline Maximum Score:", res["Baseline Maximum Score"])
        print("Baseline Difference (Max - Average):", res["Baseline Difference (Max - Average)"])
        print("Calculated (Points Mode) - Estimated Points:", res["Calculated (Points Mode) - Estimated Points"])
        print("Calculated (Points Mode) - Estimated RP:", res["Calculated (Points Mode) - Estimated RP"])
        print("Calculated (RP Mode) - Estimated Points:", res["Calculated (RP Mode) - Estimated Points"])
        print("Calculated (RP Mode) - Estimated RP:", res["Calculated (RP Mode) - Estimated RP"])
        print("--------------------------------------------------")

if __name__ == '__main__':
    main()
