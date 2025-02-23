import json
import os
import math

#########################################
# Part 1: Data Generation (Stub)
#########################################
def generate_data(filename="scouting_data_intelligent.json"):
    """
    Stub. If the file doesn't exist, generate or populate synthetic data.
    """
    pass

#########################################
# Part 2: Gather Each Team's Cycle Time
#########################################
def gather_team_cycle_data(data):
    """
    For each team t, parse:
      - autoCorals, teleCorals
      - autoAlgae, teleAlgae
      - reefRemoved
      - climbs
    Then cycUnits_i = 0.75*(aC+tC) + 1.0*(aA+tA) + 0.225*(reefRem) + 3*(climbs)
    => x_i = 150/cycUnits_i if cycUnits_i > 0 else 999
    Also track fractionMovedAuto, average endgame points, barge & proc algae, etc.
    """
    from collections import defaultdict
    end_map = {"No": 0, "Fs": 0, "Fd": 0, "P": 2, "Sc": 6, "Dc": 12}

    base = defaultdict(lambda: {
        "sumAutoC": 0.0,
        "sumTeleC": 0.0,
        "sumAutoA": 0.0,
        "sumTeleA": 0.0,
        "sumReef": 0.0,
        "sumClimbs": 0.0,
        "sumEndg": 0.0,
        "autoMoves": 0,
        "matches": 0,
        "sumBarge": 0.0,
        "sumProc": 0.0
    })

    for row in data:
        if row.get("noShow"):
            continue
        t = row["teamNumber"]
        base[t]["matches"] += 1

        # auto corals
        aC = (row.get("L1sc", 0) + row.get("L2sc", 0) +
              row.get("L3sc", 0) + row.get("L4sc", 0))
        # tele corals
        tC = (row.get("tL1sc", 0) + row.get("tL2sc", 0) +
              row.get("tL3sc", 0) + row.get("tL4sc", 0))
        base[t]["sumAutoC"] += aC
        base[t]["sumTeleC"] += tC

        # auto algae => ScAb+ScAp, tele => tScAb+ tScAp
        ab = row.get("ScAb", 0)
        ap = row.get("ScAp", 0)
        tb = row.get("tScAb", 0)
        tp = row.get("tScAp", 0)
        base[t]["sumAutoA"] += (ab + ap)
        base[t]["sumTeleA"] += (tb + tp)

        base[t]["sumBarge"] += (ab + tb)
        base[t]["sumProc"] += (ap + tp)

        # reef removal
        rr = (row.get("RmAr", 0) + row.get("RmAg", 0) +
              row.get("tRmAr", 0) + row.get("tRmAg", 0))
        base[t]["sumReef"] += rr

        # climbs => from epo
        ep = end_map.get(row.get("epo", "No"), 0)
        if ep >= 12:
            base[t]["sumClimbs"] += 1
        elif ep >= 6:
            base[t]["sumClimbs"] += 0.5
        base[t]["sumEndg"] += ep

        if row.get("Mved"):
            base[t]["autoMoves"] += 1

    final = {}
    for t, d in base.items():
        m = d["matches"]
        if m < 1:
            continue
        aC = d["sumAutoC"] / m
        tC = d["sumTeleC"] / m
        aA = d["sumAutoA"] / m
        tA = d["sumTeleA"] / m
        rr = d["sumReef"] / m
        cl = d["sumClimbs"] / m
        en = d["sumEndg"] / m
        fracMv = d["autoMoves"] / m
        bar = d["sumBarge"] / m
        pro = d["sumProc"] / m

        cycUnits = 0.75 * (aC + tC) + 1.0 * (aA + tA) + 0.225 * rr + 3 * cl
        if cycUnits < 1e-9:
            x_i = 999.0
        else:
            x_i = 150.0 / cycUnits
        final[t] = {
            "avgAutoCor": aC,
            "avgTeleCor": tC,
            "avgAutoAlg": aA,
            "avgTeleAlg": tA,
            "avgReef": rr,
            "avgClimb": cl,
            "avgEndg": en,
            "fracMoved": fracMv,
            "avgBarge": bar,
            "avgProc": pro,
            "x_i": x_i
        }
    return final

##############################################
# PART 3: Merge Alliance Stats + Build x
##############################################
def merge_alliance_data(allianceTeams, cycleData):
    """
    Summation approach:
     - sumAutoCor, sumTeleCor, sumBarge, sumProc, sumEndg, etc.
     - minFracMove => min
     - sumAutoAll => sum of avgAutoCor
     - Store each team's x_i for parallel time checks.
    Returns a dict capturing alliance-level stats.
    """
    merged = {
        "teams": allianceTeams,
        "sumAutoCor": 0.0,
        "sumTeleCor": 0.0,
        "sumBarge": 0.0,
        "sumProc": 0.0,
        "sumEndg": 0.0,
        "minFracMove": 1.0,
        "sumAutoAll": 0.0,  # total auto cor across all
        "teamXs": [],       # list of each team's x_i
        "teamData": {}      # each team's raw stats for parallel time checks
    }
    for t in allianceTeams:
        if t not in cycleData:
            continue
        st = cycleData[t]
        merged["sumAutoCor"] += st["avgAutoCor"]
        merged["sumTeleCor"] += st["avgTeleCor"]
        merged["sumBarge"] += st["avgBarge"]
        merged["sumProc"] += st["avgProc"]
        merged["sumEndg"] += st["avgEndg"]
        if st["fracMoved"] < merged["minFracMove"]:
            merged["minFracMove"] = st["fracMoved"]
        merged["sumAutoAll"] += st["avgAutoCor"]
        merged["teamXs"].append(st["x_i"])
        merged["teamData"][t] = st
    return merged

##############################################
# PART 4: Freed L2, Freed L3 enumerations
##############################################
def freed_spots(r):
    # Freed L2 and Freed L3 are limited by 2*r, capped at 12.
    return min(12, 2*r), min(12, 2*r)

##############################################
# PART 5: Shared Utils => time for alliance
##############################################
def parallel_time_for_alliance(corals, algae, reefRemoval, climbs, allianceInfo):
    """
    Given alliance-level work (corals, algae, reef removal, climbs) and each team's historical pace,
    compute the estimated match time required (using a parallel–execution approximation).
    Each team's time is calculated as:
      cyc = 0.75*(team corals) + 1.0*(team algae) + 0.225*(team reef) + 3*(team climb)
      time = cyc * x_i
    The alliance time is the maximum over teams and must be <=150s for feasibility.
    """
    totalCorHist = allianceInfo["sumAutoCor"] + allianceInfo["sumTeleCor"]
    if totalCorHist < 1e-9:
        totalCorHist = 1e-9
    totalAlgHist = allianceInfo["sumBarge"] + allianceInfo["sumProc"]
    if totalAlgHist < 1e-9:
        totalAlgHist = 1e-9
    sumReefHist = 0
    sumClimbHist = 0
    for st in allianceInfo["teamData"].values():
        sumReefHist += st["avgReef"]
        sumClimbHist += st["avgClimb"]

    if sumReefHist < 1e-9:
        sumReefHist = 1e-9
    if sumClimbHist < 1e-9:
        sumClimbHist = 1e-9

    times = []
    for st in allianceInfo["teamData"].values():
        fracC = (st["avgAutoCor"] + st["avgTeleCor"]) / totalCorHist
        usedC = fracC * corals
        fracA = (st["avgBarge"] + st["avgProc"]) / totalAlgHist
        usedAlg = fracA * algae
        fracR = st["avgReef"] / sumReefHist
        usedReef = fracR * reefRemoval
        fracCl = st["avgClimb"] / sumClimbHist
        usedCl = fracCl * climbs

        cyc = 0.75 * usedC + 1.0 * usedAlg + 0.225 * usedReef + 3.0 * usedCl
        timeSec = cyc * st["x_i"]
        times.append(timeSec)
    return max(times)

##############################################
# PART 6: Partial RP Check
##############################################
def rp_conditions(allianceInfo, c4, c3, c2, c1, bar, proc):
    """
    Return (autoRP, coralRP, bargeRP).
      - autoRP: if minFracMove >= 0.5 and sumAutoAll >= 1
      - coralRP: if each level >= 5 (or >= 3 if proc >= 2)
      - bargeRP: now only counts climb (endgame) points;
                 i.e., if sumEndg >= 14.
    """
    # auto RP
    rp_auto = 1 if (allianceInfo["minFracMove"] >= 0.5 and allianceInfo["sumAutoAll"] >= 1) else 0
    
    # coral RP
    thresh = 3 if proc >= 2 else 5
    rp_coral = 1 if (c4 >= thresh and c3 >= thresh and c2 >= thresh and c1 >= thresh) else 0
    
    # barge RP now counts only climb/endgame points
    rp_barge = 1 if (allianceInfo["sumEndg"] >= 14) else 0
    
    return rp_auto, rp_coral, rp_barge

##############################################
# PART 7: Strategy Enumerations
##############################################
def alliance_score_points(allianceInfo):
    """
    Enumerate strategies for a points-focused alliance:
      - For each R in 0..6 (reef removal cycles), compute distribution:
          * Freed L2 and L3 capacities
          * Allocate corals top-down (c4, then c3, then c2, then c1)
          * Place up to 9 algae into processor for maximum points
          * Check that the parallel time <= 150s
      - Choose the distribution with the highest points, tie-breaking on partial RP.
    Returns (points, partialRP, bestDist).
    """
    c_tot = allianceInfo["sumAutoCor"] + allianceInfo["sumTeleCor"]
    a_tot = allianceInfo["sumBarge"] + allianceInfo["sumProc"]
    if a_tot > 9:
        a_tot = 9
    best_pts = 0
    best_rp = 0
    best_dist = None

    for R in range(7):
        freedL2, freedL3 = freed_spots(R)
        # Allocate corals top-down
        c_left = c_tot
        c4 = min(12, c_left)
        c_left -= c4
        c3 = min(freedL3, c_left)
        c_left -= c3
        c2 = min(freedL2, c_left)
        c_left -= c2
        c1 = c_left  # no limit (or capped at 25 if desired)

        # Allocate algae: use all in processor for points
        proc = a_tot
        bar = 0

        # Check parallel time: reef removal is R, no climbs in this strategy
        usedCor = c4 + c3 + c2 + c1
        usedAlg = bar + proc
        usedReef = R
        usedClimb = 0
        allianceTime = parallel_time_for_alliance(usedCor, usedAlg, usedReef, usedClimb, allianceInfo)
        if allianceTime > 150:
            continue
        # Compute final points using a blended approach for coral scoring
        frac_auto = allianceInfo["sumAutoAll"] / usedCor if usedCor > 1e-9 else 0
        def blend(level):
            auto_mult = {"L4": 7, "L3": 6, "L2": 4, "L1": 3}
            tele_mult = {"L4": 5, "L3": 4, "L2": 3, "L1": 2}
            return frac_auto * auto_mult[level] + (1 - frac_auto) * tele_mult[level]
        c_pts = c4 * blend("L4") + c3 * blend("L3") + c2 * blend("L2") + c1 * blend("L1")
        a_pts = bar * 4 + proc * 6
        final_pts = c_pts + a_pts + allianceInfo["sumEndg"]
        # Partial RP conditions
        rp_auto, rp_coral, rp_barge = rp_conditions(allianceInfo, c4, c3, c2, c1, bar, proc)
        rp_sum = rp_auto + rp_coral + rp_barge
        # Choose best strategy
        if final_pts > best_pts or (abs(final_pts - best_pts) < 1e-9 and rp_sum > best_rp):
            best_pts = final_pts
            best_rp = rp_sum
            best_dist = {
                "RemovedAlgae": R,
                "FreedL2": freedL2, "FreedL3": freedL3,
                "c4": c4, "c3": c3, "c2": c2, "c1": c1,
                "barge": bar, "processor": proc,
                "Climbs": 0,
                "AllianceTime": allianceTime,
                "AlliancePoints": final_pts,
                "PartialRP": rp_sum
            }
    if best_dist is None:
        return 0, 0, {"error": "No feasible distribution for points."}
    return best_dist["AlliancePoints"], best_dist["PartialRP"], best_dist

def alliance_score_rp(allianceInfo):
    """
    RP-focused strategy:
      - Enumerate strategies over R (0..6) and must_proc options (0 or 2)
      - Prioritize achieving coral RP and barge RP
      - Check parallel time feasibility
      - Choose the strategy with highest partial RP, tie-breaking on points.
    Returns (points, partialRP, bestDist).
    """
    c_tot = allianceInfo["sumAutoCor"] + allianceInfo["sumTeleCor"]
    a_tot = allianceInfo["sumBarge"] + allianceInfo["sumProc"]
    if a_tot > 9:
        a_tot = 9
    best_rp = -999
    best_pts = 0
    best_dist = None

    def barge_needed(e):
        # bNeed such that sumEndg (climb points) + 4*barge meets 14; but now we ignore barge,
        # so we do not add any barge points.
        # (We simply consider that only endgame points count for RP from climbing.)
        return 0

    for R in range(7):
        freedL2, freedL3 = freed_spots(R)
        for must_proc in [0, 2]:
            thresh = 3 if must_proc >= 2 else 5
            needed_cor = 4 * thresh
            if needed_cor > c_tot + 1e-9:
                continue
            bNeed = barge_needed(allianceInfo["sumEndg"])
            # Place must_proc + bNeed in phase1; compute base time cost.
            base_time = parallel_time_for_alliance(needed_cor, (must_proc + bNeed), R, 0, allianceInfo)
            if base_time > 150:
                continue
            leftoverTime = 150 - base_time
            c_left = c_tot - needed_cor
            l4cap = max(0, 12 - thresh)
            l3cap = max(0, freedL3 - thresh)
            l2cap = max(0, freedL2 - thresh)
            cor2 = c_left
            if cor2 < 0:
                cor2 = 0
            usedCor2 = cor2
            a_left = a_tot - (must_proc + bNeed)
            if a_left < 0:
                a_left = 0
            leftoverAlg = a_left
            final_time = parallel_time_for_alliance(needed_cor + usedCor2, must_proc + bNeed + leftoverAlg, R, 0, allianceInfo)
            if final_time > 150:
                continue
            totalCorUsed = needed_cor + usedCor2
            frac_auto = allianceInfo["sumAutoAll"] / totalCorUsed if totalCorUsed > 1e-9 else 0
            # Naively allocate coral levels
            c4 = thresh; c3 = thresh; c2 = thresh; c1 = thresh
            leftoverC = usedCor2
            c4plus = min(l4cap, leftoverC)
            leftoverC -= c4plus; c4 += c4plus
            c3plus = min(l3cap, leftoverC)
            leftoverC -= c3plus; c3 += c3plus
            c2plus = min(l2cap, leftoverC)
            leftoverC -= c2plus; c2 += c2plus
            c1 += leftoverC
            bar = bNeed
            proc = must_proc + leftoverAlg
            def blend(level):
                autoMult = {"L4": 7, "L3": 6, "L2": 4, "L1": 3}
                teleMult = {"L4": 5, "L3": 4, "L2": 3, "L1": 2}
                return frac_auto * autoMult[level] + (1 - frac_auto) * teleMult[level]
            coralPts = c4 * blend("L4") + c3 * blend("L3") + c2 * blend("L2") + c1 * blend("L1")
            algaePts = bar * 4 + proc * 6
            finalPts = coralPts + algaePts + allianceInfo["sumEndg"]
            rp_auto, rp_coral, rp_barge = rp_conditions(allianceInfo, c4, c3, c2, c1, bar, proc)
            rp_sum = rp_auto + rp_coral + rp_barge
            if rp_sum > best_rp or (rp_sum == best_rp and finalPts > best_pts):
                best_rp = rp_sum
                best_pts = finalPts
                best_dist = {
                    "RemovedAlgae": R,
                    "threshold": thresh,
                    "FreedL2": freedL2, "FreedL3": freedL3,
                    "Cor4": c4, "Cor3": c3, "Cor2": c2, "Cor1": c1,
                    "barge": bar, "processor": proc,
                    "TimeUsed": final_time,
                    "Points": finalPts,
                    "PartialRP": rp_sum
                }
    if best_dist is None:
        return 0, 0, {"error": "No feasible rp distribution."}
    return best_dist["Points"], best_dist["PartialRP"], best_dist

def calculate_alliance_score(allianceInfo, strategy="points"):
    """
    If strategy=="points", use alliance_score_points;
    otherwise, use alliance_score_rp.
    """
    if strategy == "points":
        return alliance_score_points(allianceInfo)
    else:
        return alliance_score_rp(allianceInfo)

###############################################
# PART 8: Compare Two Alliances
###############################################
def calculate_match_outcome(data, allianceA, allianceB, stratA="points", stratB="points"):
    # Gather cycle data.
    cycleData = gather_team_cycle_data(data)
    # Build alliance info.
    Ainfo = merge_alliance_data(allianceA, cycleData)
    Binfo = merge_alliance_data(allianceB, cycleData)
    # Compute alliance scores.
    Apts, Arp, Adet = calculate_alliance_score(Ainfo, stratA)
    Bpts, Brp, Bdet = calculate_alliance_score(Binfo, stratB)
    # Determine win RP.
    Awin = 0
    Bwin = 0
    if abs(Apts - Bpts) < 1e-9:
        Awin = 1
        Bwin = 1
    elif Apts > Bpts:
        Awin = 3
    else:
        Bwin = 3
    A_total = Arp + Awin
    B_total = Brp + Bwin
    if A_total > 6:
        A_total = 6
    if B_total > 6:
        B_total = 6
    return {
        "AllianceA": {
            "Teams": allianceA,
            "Strategy": stratA,
            "Points": Apts,
            "PartialRP": Arp,
            "WinRP": Awin,
            "TotalRP": A_total,
            "Detail": Adet
        },
        "AllianceB": {
            "Teams": allianceB,
            "Strategy": stratB,
            "Points": Bpts,
            "PartialRP": Brp,
            "WinRP": Bwin,
            "TotalRP": B_total,
            "Detail": Bdet
        }
    }

###############################################
# PART 9: MAIN
###############################################
def main():
    data_file = "scouting_data_intelligent.json"
    if not os.path.exists(data_file):
        print("No data found, generating..")
        generate_data(data_file)
    else:
        print(f"Using existing data file: {data_file}")
    with open(data_file, "r") as f:
        data = json.load(f)

    # Define 5 matches.
    five_matches = [
        ([1, 2, 3], [4, 5, 6]),
        ([7, 8, 9], [10, 11, 12]),
        ([13, 14, 15], [16, 17, 18]),
        ([19, 20, 21], [22, 23, 24]),
        ([25, 26, 27], [28, 29, 30])
    ]
    combos = [("points", "points"), ("points", "rp"), ("rp", "points"), ("rp", "rp")]

    for idx, (A, B) in enumerate(five_matches, start=1):
        print(f"\n=== MATCH #{idx}: Alliance A={A} vs B={B} ===")
        for (sA, sB) in combos:
            out = calculate_match_outcome(data, A, B, sA, sB)
            Ares = out["AllianceA"]
            Bres = out["AllianceB"]
            print(f"  STRATEGIES => A={sA}, B={sB}")
            print(f"    Alliance A => Points={Ares['Points']:.1f}, PartialRP={Ares['PartialRP']}, WinRP={Ares['WinRP']}, TotalRP={Ares['TotalRP']}")
            print("      Detail =>", Ares["Detail"])
            print(f"    Alliance B => Points={Bres['Points']:.1f}, PartialRP={Bres['PartialRP']}, WinRP={Bres['WinRP']}, TotalRP={Bres['TotalRP']}")
            print("      Detail =>", Bres["Detail"])
            print()

if __name__ == "__main__":
    main()
