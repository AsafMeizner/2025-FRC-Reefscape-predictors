import json
import random
import time

# Helper function: partition an integer total into 'parts' nonnegative integers that sum to total.
def random_partition(total, parts):
    if parts == 1:
        return [total]
    # Pick (parts-1) random cut points in [0, total]
    cuts = sorted([random.randint(0, total) for _ in range(parts - 1)])
    values = []
    previous = 0
    for cut in cuts:
        values.append(cut - previous)
        previous = cut
    values.append(total - previous)
    return values

# --- Settings ---
NUM_TEAMS = 30
MATCHES_PER_TEAM = 10       # ideally
# Total team-match entries should be (NUM_TEAMS * MATCHES_PER_TEAM).
# But to have an integer number of matches (6 entries per match) we adjust.
total_team_entries = NUM_TEAMS * MATCHES_PER_TEAM  # 800
# We need a total that is divisible by 6. 800 mod 6 = 2, so add 4 extra entries.
total_team_entries += 4  # now 804 entries --> 804/6 = 134 matches exactly.
num_matches = total_team_entries // 6

# Create a list of team numbers with each team appearing MATCHES_PER_TEAM times,
# then add 4 extra (random) team appearances.
team_assignments = [team for team in range(1, NUM_TEAMS + 1) for _ in range(MATCHES_PER_TEAM)]
for _ in range(4):
    team_assignments.append(random.randint(1, NUM_TEAMS))
random.shuffle(team_assignments)  # randomize the schedule

# Starting timestamp (milliseconds)
current_time = int(time.time() * 1000)

# List to hold all submissions (each submission represents one robot's data for one match)
submissions = []
global_team_index = 0  # index into team_assignments

# We'll generate data match by match.
for match in range(1, num_matches + 1):
    # For each match there are 6 entries: first 3 are Alliance A, next 3 Alliance B.
    # For the alliance-level reef (L2, L3, L4), generate totals per alliance and partition them.
    alliance_data = {}  # will be keyed by alliance 'A' and 'B'
    for alliance in ['A', 'B']:
        alliance_data[alliance] = {}
        # For each level (except L1 which is unlimited)
        for level in ['L2', 'L3', 'L4']:
            # Total coral scored on this level for the alliance (cannot exceed 12).
            total_alliance = random.randint(0, 12)
            # Split the total between autonomous and teleop.
            aut_total = random.randint(0, total_alliance)
            teleop_total = total_alliance - aut_total
            # Partition each among the 3 robots.
            aut_parts = random_partition(aut_total, 3)
            teleop_parts = random_partition(teleop_total, 3)
            alliance_data[alliance][level] = {
                'aut': aut_parts,
                'teleop': teleop_parts
            }
    # Now create the six submissions for this match.
    # We'll assign scouter names as "Scouter 1" to "Scouter 6" (one per robot in the match).
    for alliance_index, alliance in enumerate(['A', 'B']):
        for robot_index in range(3):
            submission = {}
            # Pre-Match Fields
            # Use a scouter name based on slot (1-6)
            submission["scouter"] = f"Scouter {robot_index + alliance_index * 3 + 1}"
            submission["matchNumber"] = match
            submission["teamNumber"] = team_assignments[global_team_index]
            global_team_index += 1
            submission["noShow"] = random.random() < 0.05  # 5% chance

            # --- Autonomous Section ---
            submission["Mved"] = random.choice([True, False])
            # Scored Levels:
            # L1: unlimited – generate independently (range chosen arbitrarily)
            submission["L1sc"] = random.randint(0, 10)
            # For L2, L3, L4 use the alliance partitions
            for level in ['L2', 'L3', 'L4']:
                submission[f"{level}sc"] = alliance_data[alliance][level]['aut'][robot_index]
            # Missed Levels in Autonomous (small random number, independent for each level)
            submission["L1ms"] = random.randint(0, 2)
            for level in ['L2', 'L3', 'L4']:
                submission[f"{level}ms"] = random.randint(0, 2)
            # Collection (Autonomous)
            submission["InG"] = random.randint(0, 5)   # Coral Ground
            submission["InS"] = random.randint(0, 5)   # Coral Source
            submission["RmAr"] = random.randint(0, 5)  # Algae Reef
            submission["RmAg"] = random.randint(0, 5)  # Algae Ground
            # Additional autonomous counters
            submission["ScAb"] = random.randint(0, 3)  # Scored Algae in Barge
            submission["ScAp"] = random.randint(0, 3)  # Scored Algae in Processor

            # --- TeleOp Section ---
            # L1 (unlimited) is generated independently.
            submission["tL1sc"] = random.randint(0, 15)
            # For L2, L3, L4 use the alliance partitions for teleop
            for level in ['L2', 'L3', 'L4']:
                submission[f"t{level}sc"] = alliance_data[alliance][level]['teleop'][robot_index]
            # Missed Levels in TeleOp
            submission["tL1ms"] = random.randint(0, 2)
            for level in ['L2', 'L3', 'L4']:
                submission[f"t{level}ms"] = random.randint(0, 2)
            # TeleOp Collection counters
            submission["tInG"] = random.randint(0, 10)
            submission["tInS"] = random.randint(0, 10)
            submission["tRmAr"] = random.randint(0, 10)
            submission["tRmAg"] = random.randint(0, 10)
            # Additional teleop counters
            submission["tScAb"] = random.randint(0, 5)
            submission["tScAp"] = random.randint(0, 5)
            submission["Fou"] = random.randint(0, 2)  # Fouls

            # --- End Game ---
            submission["epo"] = random.choice(["No", "P", "Sc", "Dc", "Fd", "Fs"])

            # --- Post-Match ---
            submission["dto"] = random.random() < 0.1  # 10% chance robot flipped/fell over
            submission["yc"] = random.choices(["No Card", "Yellow", "Red"], weights=[90, 8, 2])[0]
            submission["co"] = random.choice([
                "", "Good performance", "Minor issues", 
                "Robot was stuck", "Exceeded expectations", "Needs improvement"
            ])

            # Simulated submission time (milliseconds)
            submission["submissionTime"] = current_time + random.randint(0, 1000000)

            submissions.append(submission)

# Write the generated submissions to a JSON file.
with open("scouting_data_intelligent.json", "w") as f:
    json.dump(submissions, f, indent=2)

print(f"Generated {len(submissions)} submissions across {num_matches} matches.")
