import json
import os
import math
import random
import numpy as np
import pandas as pd
import gym
from gym import spaces
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers  # type: ignore
from collections import defaultdict
from tqdm import trange

##############################################
# Improved Calculation Functions
##############################################

def gather_team_cycle_data(data):
    """
    Aggregate cycle data per team from raw scouting entries.
    Computes averages for auto/tele corals, algae, reef removal, climbs, etc.
    """
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
        # auto & tele corals
        aC = row.get("L1sc", 0) + row.get("L2sc", 0) + row.get("L3sc", 0) + row.get("L4sc", 0)
        tC = row.get("tL1sc", 0) + row.get("tL2sc", 0) + row.get("tL3sc", 0) + row.get("tL4sc", 0)
        base[t]["sumAutoC"] += aC
        base[t]["sumTeleC"] += tC
        # auto & tele algae (barge: ScAb/ tScAb; processor: ScAp/ tScAp)
        ab = row.get("ScAb", 0)
        ap = row.get("ScAp", 0)
        tb = row.get("tScAb", 0)
        tp = row.get("tScAp", 0)
        base[t]["sumAutoA"] += (ab + ap)
        base[t]["sumTeleA"] += (tb + tp)
        base[t]["sumBarge"] += (ab + tb)
        base[t]["sumProc"] += (ap + tp)
        # reef removal
        rr = row.get("RmAr", 0) + row.get("RmAg", 0) + row.get("tRmAr", 0) + row.get("tRmAg", 0)
        base[t]["sumReef"] += rr
        # climbs/endgame from 'epo'
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
        x_i = 150.0 / cycUnits if cycUnits > 1e-9 else 999.0
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

def merge_alliance_data(allianceTeams, cycleData):
    """
    Merge the cycle data for a set of alliance teams.
    Returns a dict with aggregated stats needed for later calculations.
    """
    merged = {
      "teams": allianceTeams,
      "sumAutoCor": 0.0,
      "sumTeleCor": 0.0,
      "sumBarge": 0.0,
      "sumProc": 0.0,
      "sumEndg": 0.0,
      "minFracMove": 1.0,
      "sumAutoAll": 0.0,  # total auto corals (used for auto fraction)
      "teamXs": [],
      "teamData": {}
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

def freed_spots(r):
    # Freed L2 and L3 are limited by 2*r, capped at 12.
    return min(12, 2*r), min(12, 2*r)

def parallel_time_for_alliance(corals, algae, reefRemoval, climbs, allianceInfo):
    """
    Given the alliance’s total work (corals, algae, reef removal, climbs) and each team's historical pace,
    compute the estimated match time needed (using a parallel–execution approximation).
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

def rp_conditions(allianceInfo, c4, c3, c2, c1, bar, proc):
    """
    Determine the partial ranking points (RP) conditions.
      - autoRP: if minimum fraction moved ≥ 0.5 and there is some auto coral
      - coralRP: if each level’s count is ≥ threshold (3 if proc ≥ 2, else 5)
      - bargeRP: now counts only the endgame (climb) points; i.e., if sumEndg ≥ 14.
    """
    rp_auto = 1 if (allianceInfo["minFracMove"] >= 0.5 and (allianceInfo["sumAutoCor"] > 0)) else 0
    thresh = 3 if proc >= 2 else 5
    rp_coral = 1 if (c4 >= thresh and c3 >= thresh and c2 >= thresh and c1 >= thresh) else 0
    rp_barge = 1 if (allianceInfo["sumEndg"] >= 14) else 0
    return rp_auto, rp_coral, rp_barge

##############################################
# New Score Functions Parameterized by Agent Action
##############################################

def compute_alliance_score_points(allianceInfo, R):
    """
    For a points-focused strategy, use the chosen R (reef removal cycles: 0-6)
    to allocate corals and algae (all processor) and compute the final predicted score.
    Returns (final_points, partial_RP, detail_dict).
    """
    c_tot = allianceInfo["sumAutoCor"] + allianceInfo["sumTeleCor"]
    a_tot = allianceInfo["sumBarge"] + allianceInfo["sumProc"]
    if a_tot > 9:
        a_tot = 9
    freedL2, freedL3 = freed_spots(R)
    c_left = c_tot
    c4 = min(12, c_left)
    c_left -= c4
    c3 = min(freedL3, c_left)
    c_left -= c3
    c2 = min(freedL2, c_left)
    c_left -= c2
    c1 = c_left
    proc = a_tot  # all algae into processor for maximum points
    bar = 0
    usedCor = c4 + c3 + c2 + c1
    usedAlg = proc  # bar is 0
    usedReef = R
    usedClimb = 0
    allianceTime = parallel_time_for_alliance(usedCor, usedAlg, usedReef, usedClimb, allianceInfo)
    if allianceTime > 150:
        return -100, 0, {"AllianceTime": allianceTime}
    frac_auto = allianceInfo["sumAutoAll"] / usedCor if usedCor > 1e-9 else 0
    def blend(level):
        auto_mult = {"L4": 7, "L3": 6, "L2": 4, "L1": 3}
        tele_mult = {"L4": 5, "L3": 4, "L2": 3, "L1": 2}
        return frac_auto * auto_mult[level] + (1 - frac_auto) * tele_mult[level]
    c_pts = c4 * blend("L4") + c3 * blend("L3") + c2 * blend("L2") + c1 * blend("L1")
    a_pts = proc * 6  # processor gives 6 points per unit
    final_pts = c_pts + a_pts + allianceInfo["sumEndg"]
    rp_auto, rp_coral, rp_barge = rp_conditions(allianceInfo, c4, c3, c2, c1, bar, proc)
    rp_sum = rp_auto + rp_coral + rp_barge
    detail = {"c4": c4, "c3": c3, "c2": c2, "c1": c1, "proc": proc, "R": R, "AllianceTime": allianceTime}
    return final_pts, rp_sum, detail

def compute_alliance_score_rp(allianceInfo, R, must_proc):
    """
    For an RP-focused strategy, the agent’s action specifies R (0-6) and a "must_proc" value (0 or 2).
    Returns (final_points, partial_RP, detail_dict).
    """
    c_tot = allianceInfo["sumAutoCor"] + allianceInfo["sumTeleCor"]
    a_tot = allianceInfo["sumBarge"] + allianceInfo["sumProc"]
    if a_tot > 9:
        a_tot = 9
    thresh = 3 if must_proc >= 2 else 5
    needed_cor = 4 * thresh
    if needed_cor > c_tot:
        return -100, -100, {"error": "Not enough corals for RP"}
    def barge_needed(e):
        return max(0, math.ceil((14 - e) / 4))
    bNeed = barge_needed(allianceInfo["sumEndg"])
    base_time = parallel_time_for_alliance(needed_cor, (must_proc + bNeed), R, 0, allianceInfo)
    if base_time > 150:
        return -100, -100, {"AllianceTime": base_time}
    usedCor2 = c_tot - needed_cor
    totalCorUsed = needed_cor + usedCor2
    frac_auto = allianceInfo["sumAutoAll"] / totalCorUsed if totalCorUsed > 1e-9 else 0
    # For simplicity, distribute the needed coral equally and add leftover to L4
    c4 = thresh + usedCor2
    c3 = thresh
    c2 = thresh
    c1 = thresh
    a_left = a_tot - (must_proc + bNeed)
    if a_left < 0:
        a_left = 0
    bar = bNeed
    proc = must_proc + a_left
    usedAlg = bar + proc
    usedReef = R
    usedClimb = 0
    final_time = parallel_time_for_alliance(totalCorUsed, usedAlg, R, 0, allianceInfo)
    if final_time > 150:
        return -100, -100, {"AllianceTime": final_time}
    def blend(level):
        autoMult = {"L4": 7, "L3": 6, "L2": 4, "L1": 3}
        teleMult = {"L4": 5, "L3": 4, "L2": 3, "L1": 2}
        return frac_auto * autoMult[level] + (1 - frac_auto) * teleMult[level]
    coralPts = c4 * blend("L4") + c3 * blend("L3") + c2 * blend("L2") + c1 * blend("L1")
    algaePts = bar * 4 + proc * 6
    finalPts = coralPts + algaePts + allianceInfo["sumEndg"]
    rp_auto, rp_coral, rp_barge = rp_conditions(allianceInfo, c4, c3, c2, c1, bar, proc)
    rp_sum = rp_auto + rp_coral + rp_barge
    detail = {"c4": c4, "c3": c3, "c2": c2, "c1": c1, "bar": bar, "proc": proc, "R": R, "must_proc": must_proc, "AllianceTime": final_time}
    return finalPts, rp_sum, detail

##############################################
# New Gym Environment: ReefscapeCalcEnv
##############################################

class ReefscapeCalcEnv(gym.Env):
    """
    A one-step environment where the state is the aggregated alliance info (from historical cycle data)
    and the agent selects a strategy parameter.
    
    For focus 'points': action ∈ {0, 1, ..., 6} representing R (reef removal cycles).
    For focus 'rp': action ∈ {0, 1, ..., 13} where we decode:
         R = action // 2   (0-6)
         must_proc = 0 if action is even, else 2.
    The reward is the predicted final alliance score (and partial RP can be used as a bonus).
    """
    def __init__(self, allianceTeams, cycleData, focus='points'):
        super(ReefscapeCalcEnv, self).__init__()
        self.allianceTeams = allianceTeams
        self.cycleData = cycleData
        self.allianceInfo = merge_alliance_data(allianceTeams, cycleData)
        self.focus = focus
        # Define state as an 8-dimensional vector of key stats.
        # [sumAutoCor, sumTeleCor, sumBarge, sumProc, sumEndg, minFracMove, sumAutoAll, mean(x_i)]
        mean_x = np.mean(self.allianceInfo["teamXs"]) if self.allianceInfo["teamXs"] else 0.0
        self.state = np.array([
            self.allianceInfo["sumAutoCor"],
            self.allianceInfo["sumTeleCor"],
            self.allianceInfo["sumBarge"],
            self.allianceInfo["sumProc"],
            self.allianceInfo["sumEndg"],
            self.allianceInfo["minFracMove"],
            self.allianceInfo["sumAutoAll"],
            mean_x
        ], dtype=np.float32)
        # Action space
        if self.focus == "points":
            self.action_space = spaces.Discrete(7)  # R=0..6
        else:
            self.action_space = spaces.Discrete(14) # 7 values for R * 2 options for must_proc (0 or 2)
        # Observation space is a Box with our 8 features.
        self.observation_space = spaces.Box(low=0, high=999, shape=(8,), dtype=np.float32)
        self.done = False

    def reset(self):
        self.done = False
        return self.state

    def step(self, action):
        if self.done:
            return self.state, 0, self.done, {}
        # Decode action and compute predicted score using improved formulas.
        if self.focus == "points":
            R = action  # 0 to 6
            final_pts, rp_sum, detail = compute_alliance_score_points(self.allianceInfo, R)
            reward = final_pts + rp_sum * 2  # example: bonus for RP (tweak as desired)
        else:
            # action in 0..13: decode into R and must_proc.
            R = action // 2  # integer division: 0..6
            must_proc = 0 if action % 2 == 0 else 2
            final_pts, rp_sum, detail = compute_alliance_score_rp(self.allianceInfo, R, must_proc)
            reward = final_pts + rp_sum * 100  # heavier bonus on RP
        self.done = True
        info = {"detail": detail, "focus": self.focus}
        return self.state, reward, self.done, info

##############################################
# DQN Agent (adapted for one–step episodes)
##############################################

class DQNAgent:
    def __init__(self, obs_size, num_actions, lr=1e-3, gamma=0.99):
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.lr = lr
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target()
        self.replay_buffer = []
        self.buffer_size = 4000
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def build_model(self):
        inputs = layers.Input(shape=(self.obs_size,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(128, activation='relu')(x)
        # Value stream
        value = layers.Dense(64, activation='relu')(x)
        value = layers.Dense(1)(value)
        # Advantage stream
        advantage = layers.Dense(64, activation='relu')(x)
        advantage = layers.Dense(self.num_actions)(advantage)
        advantage_mean = layers.Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
        q_values = layers.Add()([value, layers.Subtract()([advantage, advantage_mean])])
        model = tf.keras.Model(inputs=inputs, outputs=q_values)
        model.compile(optimizer=optimizers.Adam(self.lr), loss='mse')
        return model

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(np.array([state]), verbose=0)[0]
            if done:
                target[action] = reward
            else:
                t = self.target_model.predict(np.array([next_state]), verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)
            states.append(state)
            targets.append(target)
        self.model.train_on_batch(np.array(states), np.array(targets))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

##############################################
# Training & Evaluation Routines
##############################################

def sample_alliance_env(cycleData, focus='points'):
    # Randomly sample 3 teams from the cycleData keys.
    all_teams = list(cycleData.keys())
    alliance = random.sample(all_teams, 3)
    env = ReefscapeCalcEnv(alliance, cycleData, focus=focus)
    return env, alliance

def train_agent(cycleData, focus, agent, episodes=1000):
    for e in trange(episodes, desc=f"Training DQN ({focus})"):
        env, _ = sample_alliance_env(cycleData, focus=focus)
        state = env.reset()
        # Since it's one-step, get one transition:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        if e % 100 == 0:
            agent.update_target()
            print(f"[{focus.upper()}] Episode {e}, Reward={reward:.1f}, Info={info}, Epsilon={agent.epsilon:.3f}")

def evaluate_agent(cycleData, focus, agent):
    env, alliance = sample_alliance_env(cycleData, focus=focus)
    state = env.reset()
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    print(f"\n=== EVALUATION ({focus.upper()} Focus) ===")
    print("Alliance Teams:", alliance)
    print("Chosen Action:", action)
    if focus == "points":
        print("Interpreted R =", action)
    else:
        R = action // 2
        must_proc = 0 if action % 2 == 0 else 2
        print("Interpreted R =", R, "and must_proc =", must_proc)
    print("Predicted Alliance Score (reward):", reward)
    print("Detail:", info["detail"])

##############################################
# MAIN: Load Data, Compute Cycle Data, and Train Agents
##############################################

def main():
    data_file = "scouting_data_intelligent.json"
    if not os.path.exists(data_file):
        print("Data file not found. Please generate scouting data first.")
        return
    else:
        print("Using existing data file:", data_file)
    with open(data_file, "r") as f:
        data = json.load(f)
    # Compute cycle data from raw scouting entries.
    cycleData = gather_team_cycle_data(data)
    
    # Define observation dimension (8 features, as set in ReefscapeCalcEnv)
    obs_size = 8
    # For points focus: 7 discrete actions; for rp: 14.
    num_actions_points = 7
    num_actions_rp = 14

    agent_points = DQNAgent(obs_size, num_actions_points, lr=1e-3)
    agent_rp = DQNAgent(obs_size, num_actions_rp, lr=1e-3)

    print("\n=== Training DQN Agent (Points Focus) ===")
    train_agent(cycleData, 'points', agent_points, episodes=1000)
    evaluate_agent(cycleData, 'points', agent_points)
    agent_points.model.save("dqn_points_improved.h5")
    print("Saved improved points model as dqn_points_improved.h5")

    print("\n=== Training DQN Agent (RP Focus) ===")
    train_agent(cycleData, 'rp', agent_rp, episodes=1000)
    evaluate_agent(cycleData, 'rp', agent_rp)
    agent_rp.model.save("dqn_rp_improved.h5")
    print("Saved improved RP model as dqn_rp_improved.h5")

if __name__ == '__main__':
    main()
