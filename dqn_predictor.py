import json
import random
import time
import os
import math
import numpy as np
import pandas as pd
import gym
from gym import spaces
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tqdm import trange

#########################################
# Part 1: Data Generation
#########################################
def generate_data(filename="scouting_data_intelligent.json"):
    """
    Generate synthetic match data for Reefscape, with 70 teams * 30 matches each.
    Each alliance can score up to 12 corals per level; algae totals up to 36.
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

    NUM_TEAMS = 70
    MATCHES_PER_TEAM = 30
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
                total_alliance = random.randint(0, 12)
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

                # Autonomous
                submission["Mved"] = random.choice([True, False])
                submission["L1sc"] = random.randint(0, 10)  
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

                # TeleOp
                submission["tL1sc"] = random.randint(0, 15)
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

                # End Game
                submission["epo"] = random.choice(["No", "P", "Sc", "Dc", "Fd", "Fs"])

                # Post-Match
                submission["dto"] = random.random() < 0.1
                submission["yc"] = random.choices(["No Card", "Yellow", "Red"], weights=[90,8,2])[0]
                submission["co"] = random.choice(["","Good performance","Minor issues","Robot was stuck","Exceeded expectations","Needs improvement"])
                submission["submissionTime"] = current_time + random.randint(0, 1000000)
                submissions.append(submission)

    with open(filename, "w") as f:
        json.dump(submissions, f, indent=2)
    print(f"Generated", len(submissions), "submissions across", num_matches, "matches.")

#########################################
# Part 2: Supporting Calculation Functions
#########################################
def average_auto_score(data, teamNumber):
    """Compute average auto coral using multipliers L1=3, L2=4, L3=6, L4=7."""
    teamData = [entry for entry in data if entry["teamNumber"] == teamNumber]
    if not teamData:
        return 0
    total = sum(entry.get("L1sc",0)*3 + entry.get("L2sc",0)*4 +
                entry.get("L3sc",0)*6 + entry.get("L4sc",0)*7
                for entry in teamData)
    return total / len(teamData)

def average_teleop_score(data, teamNumber):
    """Compute average teleop coral using multipliers L1=2, L2=3, L3=4, L4=5."""
    teamData = [entry for entry in data if entry["teamNumber"] == teamNumber]
    if not teamData:
        return 0
    total = sum(entry.get("tL1sc",0)*2 + entry.get("tL2sc",0)*3 +
                entry.get("tL3sc",0)*4 + entry.get("tL4sc",0)*5
                for entry in teamData)
    return total / len(teamData)

def average_algae_score(data, teamNumber):
    """Compute average algae from barge=4 and processor=6."""
    teamData = [entry for entry in data if entry["teamNumber"] == teamNumber]
    if not teamData:
        return 0
    total = sum(
        entry.get("ScAb",0)*4 + entry.get("ScAp",0)*6 +
        entry.get("tScAb",0)*4 + entry.get("tScAp",0)*6
        for entry in teamData
    )
    return total / len(teamData)

def average_endgame_points(data, teamNumber):
    """Average endgame points from 'epo' field."""
    mapping = {"No":0,"P":2,"Sc":6,"Dc":12,"Fd":0,"Fs":0}
    teamData = [entry for entry in data if entry["teamNumber"] == teamNumber]
    if not teamData:
        return 0
    total = sum(mapping.get(entry.get("epo","No"),0) for entry in teamData)
    return total / len(teamData)

def calculate_estimated_alliance_score(data, team_numbers):
    """Reference approach: sum average scores and compute simple RP (max=6)."""
    total_score = 0
    for t in team_numbers:
        total_score += average_auto_score(data, t)
        total_score += average_teleop_score(data, t)
        total_score += average_algae_score(data, t)
        total_score += average_endgame_points(data, t)

    # Estimate RP
    rp = 0
    # Coral: if for each level (L1-L4) we average≥5
    levels = ["L1","L2","L3","L4"]
    for level in levels:
        vals=[]
        for t in team_numbers:
            teamData = [e for e in data if e["teamNumber"]==t]
            if teamData:
                avg = sum(e.get(level+"sc",0)+e.get("t"+level+"sc",0) for e in teamData)/len(teamData)
                vals.append(avg)
        if vals and sum(vals)/len(vals)>=5:
            rp+=1
    # Auto: all teams mved≥50% and scored≥1 auto coral
    def avg_Mved(t):
        tData=[e for e in data if e["teamNumber"]==t]
        if not tData:
            return 0
        return sum(1 if e.get("Mved",False) else 0 for e in tData)/len(tData)
    if all(avg_Mved(t)>=0.5 for t in team_numbers):
        rp+=1
    # Barge≥14
    def avg_algae_barge(t):
        tData=[e for e in data if e["teamNumber"]==t]
        if not tData: return 0
        return sum(e.get("ScAb",0)*4 + e.get("tScAb",0)*4 for e in tData)/len(tData)
    barge= sum(avg_algae_barge(t) for t in team_numbers)/ len(team_numbers)
    if barge>=14: rp+=1
    # Win=+3
    rp+=3
    if rp>6: rp=6

    return total_score, rp

#########################################
# Part 3: Enhanced ReefscapeEnv with Reward Shaping
#########################################
class ReefscapeEnv(gym.Env):
    """
    Enhanced environment that provides different reward shaping depending on focus:
      - focus='points': immediate reward = points
      - focus='rp': smaller immediate reward, large final reward for each RP
    """
    def __init__(self, team_stats, focus='points'):
        super(ReefscapeEnv, self).__init__()
        self.team_stats = team_stats
        self.focus = focus
        self.total_time = 150.0
        self.auto_time = 15.0
        self.time = 0.0
        self.phase = 'auto'
        self.num_teams = len(team_stats)
        self.coral_remaining = {'L1':12,'L2':12,'L3':12,'L4':12}
        self.algae_remaining = 36
        self.climb_status = {team: None for team in self.team_stats}
        self.alliance_points = 0.0
        self.alliance_rp = 0
        self.done=False

        # 13 actions per robot => 13^num_teams joint actions
        self.action_space = spaces.MultiDiscrete([13]*self.num_teams)
        obs_low = np.zeros(1+4+1+self.num_teams)
        obs_high= np.array([self.total_time,12,12,12,12,36]+[1]*self.num_teams,dtype=np.float32)
        self.observation_space=spaces.Box(low=obs_low,high=obs_high,dtype=np.float32)

    def _get_obs(self):
        climb_flags=[1 if self.climb_status[team] else 0 for team in self.team_stats]
        return np.array([self.total_time-self.time,
                         self.coral_remaining['L1'],
                         self.coral_remaining['L2'],
                         self.coral_remaining['L3'],
                         self.coral_remaining['L4'],
                         self.algae_remaining]+ climb_flags, dtype=np.float32)

    def step(self, actions):
        if self.time< self.auto_time:
            self.phase='auto'
        elif self.time< self.total_time-10:
            self.phase='teleop'
        else:
            self.phase='endgame'
        reward=0.0

        for i, team in enumerate(list(self.team_stats.keys())):
            act= actions[i]
            dt= 1.0
            if self.phase in['auto','teleop']:
                if act==0:
                    dt=1.0
                elif act in[1,2,3,4]:
                    level={1:'L1',2:'L2',3:'L3',4:'L4'}[act]
                    if self.coral_remaining[level]>0:
                        key= f"{self.phase}_coral_{level}"
                        success_rate=self.team_stats[team].get(key+'_rate',0.8)
                        cycle_time=self.team_stats[team].get(key+'_cycle',10.0)
                        if np.random.rand()<success_rate:
                            self.coral_remaining[level]-=1
                            pts_tuple=(3,4,6,7) if self.phase=='auto' else(2,3,4,5)
                            pts=pts_tuple[act-1]
                            self.alliance_points += pts
                            if self.focus=='points':
                                reward+=pts
                            else:
                                reward+=pts*0.1
                        dt=cycle_time
                    else:
                        dt=1.0
                elif act==5:
                    if self.algae_remaining>0:
                        key= f"{self.phase}_algae_barge"
                        success_rate=self.team_stats[team].get(key+'_rate',0.8)
                        cycle_time=self.team_stats[team].get(key+'_cycle',10.0)
                        if np.random.rand()<success_rate:
                            self.algae_remaining-=1
                            pts=4
                            self.alliance_points+=pts
                            if self.focus=='points':
                                reward+=pts
                            else:
                                reward+=pts*0.1
                        dt=cycle_time
                    else:
                        dt=1.0
                elif act==6:
                    key= f"{self.phase}_algae_processor"
                    success_rate= self.team_stats[team].get(key+'_rate',0.8)
                    cycle_time= self.team_stats[team].get(key+'_cycle',10.0)
                    if np.random.rand()<success_rate:
                        pts=6
                        self.alliance_points+=pts
                        if self.focus=='points':
                            reward+=pts
                        else:
                            reward+=pts*0.1
                    dt=cycle_time
                elif act in[7,8]:
                    dt=5.0
                else:
                    dt=1.0
            else:# endgame
                if act<9:
                    act=9
                if self.climb_status[team] is None:
                    if act==9:
                        pts=0
                    elif act==10:
                        pts=2
                    elif act==11:
                        pts=6
                    elif act==12:
                        pts=12
                    self.climb_status[team]= act
                    self.alliance_points+=pts
                    if self.focus=='points':
                        reward+=pts
                    else:
                        reward+=pts*0.1
                dt=0
            self.time+=dt
            if self.time>=self.total_time:
                self.done=True

        if self.phase=='endgame':
            self.time=self.total_time
            self.done=True

        if self.done:
            coral_scored={lvl:(12-self.coral_remaining[lvl])for lvl in self.coral_remaining}
            rp=0
            if all(coral_scored[lvl]>=5 for lvl in coral_scored):
                rp+=1
            all_auto= all(self.team_stats[team].get('auto_left',1.0)>=0.5 for team in self.team_stats)
            if all_auto:
                rp+=1
            barge_points=(36-self.algae_remaining)*4
            if barge_points>=14:
                rp+=1
            rp+=3
            if rp>6: rp=6
            self.alliance_rp=rp
            if self.focus=='rp':
                # big final reward for RP, small bonus for points
                reward+= rp*1000
                reward+= self.alliance_points*0.1
            else:
                # if focusing on points, small final reward for rp
                reward+=rp*2

        return self._get_obs(), reward, self.done, {"points":self.alliance_points,"rp":self.alliance_rp}

    def reset(self):
        self.time=0.0
        self.phase='auto'
        self.coral_remaining={'L1':12,'L2':12,'L3':12,'L4':12}
        self.algae_remaining=36
        self.climb_status={team:None for team in self.team_stats}
        self.alliance_points=0.0
        self.alliance_rp=0
        self.done=False
        return self._get_obs()

#########################################
# Part 4: DQN Agent
#########################################
NUM_ACTIONS_PER_TEAM=13

def decode_action(action_int,num_teams=3,num_actions=NUM_ACTIONS_PER_TEAM):
    actions=[]
    for _ in range(num_teams):
        actions.append(action_int%num_actions)
        action_int//=num_actions
    return actions

class DQNAgent:
    def __init__(self,obs_size,num_actions,lr=1e-3,gamma=0.99):
        self.obs_size=obs_size
        self.num_actions=num_actions
        self.gamma=gamma
        self.lr=lr
        self.model= self.build_model()
        self.target_model= self.build_model()
        self.update_target()
        self.replay_buffer=[]
        self.buffer_size=40000
        self.batch_size=128
        self.epsilon=1.0
        self.epsilon_min=0.05
        self.epsilon_decay=0.995

    def build_model(self):
        inputs= layers.Input(shape=(self.obs_size,))
        x= layers.Dense(256,activation='relu')(inputs)
        x= layers.Dense(256,activation='relu')(x)
        # Value stream
        value= layers.Dense(128,activation='relu')(x)
        value= layers.Dense(1)(value)
        # Advantage stream
        advantage= layers.Dense(128,activation='relu')(x)
        advantage= layers.Dense(self.num_actions)(advantage)
        advantage_mean= layers.Lambda(lambda a: tf.reduce_mean(a,axis=1,keepdims=True))(advantage)
        q_values= layers.Add()([value, layers.Subtract()([advantage, advantage_mean])])
        model= tf.keras.Model(inputs=inputs, outputs=q_values)
        model.compile(optimizer=optimizers.Adam(self.lr), loss='mse')
        return model

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand()<self.epsilon:
            return np.random.randint(self.num_actions)
        q_values= self.model.predict(np.array([state]),verbose=0)[0]
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state,action,reward,next_state,done))
        if len(self.replay_buffer)> self.buffer_size:
            self.replay_buffer.pop(0)

    def replay(self):
        if len(self.replay_buffer)< self.batch_size:
            return
        minibatch_indices= np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        states, targets= [],[]
        for idx in minibatch_indices:
            state,action,reward,next_state,done= self.replay_buffer[idx]
            target= self.model.predict(np.array([state]),verbose=0)[0]
            if done:
                target[action]= reward
            else:
                t= self.target_model.predict(np.array([next_state]),verbose=0)[0]
                target[action]= reward+ self.gamma* np.amax(t)
            states.append(state)
            targets.append(target)
        self.model.train_on_batch(np.array(states), np.array(targets))
        if self.epsilon> self.epsilon_min:
            self.epsilon*= self.epsilon_decay

#########################################
# Part 5: Training & Evaluation
#########################################
def sample_alliance_env(team_stats_pool, focus='points'):
    alliance_team_numbers= random.sample(list(team_stats_pool.keys()),3)
    alliance_team_stats={team:team_stats_pool[team] for team in alliance_team_numbers}
    env= ReefscapeEnv(alliance_team_stats,focus=focus)
    return env, alliance_team_numbers

def train_agent(team_stats_pool,focus,agent,episodes=2000):
    for e in trange(episodes, desc=f"Training DQN ({focus})"):
        env, _ = sample_alliance_env(team_stats_pool,focus=focus)
        state= env.reset()
        total_reward=0
        while True:
            joint_action_int= agent.act(state)
            actions= decode_action(joint_action_int,num_teams=env.num_teams,num_actions=NUM_ACTIONS_PER_TEAM)
            next_state, reward, done, info= env.step(actions)
            agent.remember(state,joint_action_int,reward,next_state,done)
            state= next_state
            total_reward+= reward
            if done: break
        agent.replay()
        if e % 100==0:
            agent.update_target()
            print(f"[{focus.upper()}] Episode {e}, Reward={total_reward:.1f}, Points={info['points']}, RP={info['rp']}, Epsilon={agent.epsilon:.3f}")

def evaluate_agent(team_stats_pool,focus,agent):
    env,alliance_team_numbers= sample_alliance_env(team_stats_pool,focus=focus)
    state= env.reset()
    done=False
    actions_taken=[]
    while not done:
        joint_action_int= agent.act(state)
        actions= decode_action(joint_action_int,num_teams=env.num_teams,num_actions=NUM_ACTIONS_PER_TEAM)
        actions_taken.append(actions)
        state, reward, done, info= env.step(actions)
    print(f"\n=== EVALUATION ({focus.upper()} Focus) ===")
    print("Alliance Teams:",alliance_team_numbers)
    print("Final Alliance Points:",info["points"])
    print("Final Alliance RP:",info["rp"])
    print("Sequence of Joint Actions:")
    for a in actions_taken:
        print(a)

def main():
    data_file= "scouting_data_intelligent.json"
    if not os.path.exists(data_file):
        print("Generating synthetic data...")
        generate_data(data_file)
    else:
        print("Using existing data file:",data_file)
    with open(data_file,"r") as f:
        data= json.load(f)

    df= pd.DataFrame(data)
    agg_df= df.groupby("teamNumber").agg({
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
        total= sc+ ms
        return sc/total if total>0 else 0.5

    team_stats_pool={}
    for team, row in agg_df.iterrows():
        team_stats_pool[team]= {
            'auto_coral_L1_rate': compute_rate(row['L1sc'],row['L1ms']),
            'auto_coral_L1_cycle': 8.0,
            'auto_coral_L2_rate': compute_rate(row['L2sc'],row['L2ms']),
            'auto_coral_L2_cycle': 10.0,
            'auto_coral_L3_rate': compute_rate(row['L3sc'],row['L3ms']),
            'auto_coral_L3_cycle': 12.0,
            'auto_coral_L4_rate': compute_rate(row['L4sc'],row['L4ms']),
            'auto_coral_L4_cycle': 15.0,
            'teleop_coral_L1_rate': compute_rate(row['tL1sc'],row['tL1ms']),
            'teleop_coral_L1_cycle': 6.0,
            'teleop_coral_L2_rate': compute_rate(row['tL2sc'],row['tL2ms']),
            'teleop_coral_L2_cycle': 8.0,
            'teleop_coral_L3_rate': compute_rate(row['tL3sc'],row['tL3ms']),
            'teleop_coral_L3_cycle': 10.0,
            'teleop_coral_L4_rate': compute_rate(row['tL4sc'],row['tL4ms']),
            'teleop_coral_L4_cycle': 12.0,
            'auto_algae_barge_rate': 0.8, 'auto_algae_barge_cycle':8.0,
            'auto_algae_processor_rate': 0.7, 'auto_algae_processor_cycle':8.0,
            'teleop_algae_barge_rate': 0.85, 'teleop_algae_barge_cycle':6.0,
            'teleop_algae_processor_rate': 0.75, 'teleop_algae_processor_cycle':6.0,
            'auto_left':1.0 if row['Mved']>=0.5 else 0.0
        }

    # Calculation-based approach
    alliance= [20,11,28]
    calc_points,calc_rp= calculate_estimated_alliance_score(data,alliance)
    print("\n=== Calculation-Based Alliance Prediction ===")
    print("Alliance Teams:",alliance)
    print("Estimated Total Points:",round(calc_points,2))
    print("Estimated RP:",calc_rp)

    # DQN agents
    obs_size=9
    num_joint_actions= NUM_ACTIONS_PER_TEAM**3

    agent_points= DQNAgent(obs_size,num_joint_actions,lr=1e-3)
    agent_rp= DQNAgent(obs_size,num_joint_actions,lr=1e-3)

    # Train Points Focus
    print("\n=== Training DQN Agent (Points Focus) ===")
    train_agent(team_stats_pool,'points',agent_points,episodes=2000)
    evaluate_agent(team_stats_pool,'points',agent_points)
    agent_points.model.save("dqn_points.h5")
    print("Saved Points model as dqn_points.h5")

    # Train RP Focus
    print("\n=== Training DQN Agent (RP Focus) ===")
    train_agent(team_stats_pool,'rp',agent_rp,episodes=2000)
    evaluate_agent(team_stats_pool,'rp',agent_rp)
    agent_rp.model.save("dqn_rp.h5")
    print("Saved RP model as dqn_rp.h5")

if __name__=='__main__':
    main()
