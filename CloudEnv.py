import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import signal
import numpy as np
import pyglet
from arbi_agent.model import generalized_list_factory as GLFactory
from random import *


class CloudWorldEnvImage(gym.Env):

    #goal types:  stack, stack top blue, stack buttom blue
    def __init__(self, goal_type='cloud',reward=1.0, penalty=-.1,error_penalty=-.1):
        
        self.predicates = list()
        self.action_space = spaces.Discrete(4*4) #
        self.goal_type=goal_type
        self.seed()
        self.penalty=penalty
        self.error_penalty=error_penalty
        self.reward=reward
        self.episode=0

    def get_randState(self):
        pred = [0, 0, 0, 0, 0]
        r = randint(0, 4)
        pred[r] = 1
        return pred
        
    def get_b_s(self, preds):
        b_s = [0, 0, 0, 0, 0]

        for pred in preds:
            pred = pred.split()
            print(pred)
            if pred[0] == "batteryRemain":
                if int(pred[2]) > 80:
                    b_s[4] = 1
                elif int(pred[2]) > 60:
                    b_s[3] = 1 
                elif int(pred[2]) > 40:
                    b_s[2] = 1 
                elif int(pred[2]) > 20:
                    b_s[1] = 1 
                else:
                    b_s[0] = 1 
        return b_s
        
    def get_c_s(self, preds):
        c_s = [0, 0, 0, 0, 0]
        cnt = 4
        for pred in preds:
            pred = pred.split()
            if pred[0] == "collidable":
                cnt = cnt - 1
        if cnt<0: 
            cnt=0             
        c_s[cnt] = 1
        
        return c_s
        
    def get_s_s(self, preds):
        s_s = [0, 0, 1, 0, 0]
        for pred in preds:
            pred = pred.split()
            if pred[0] == "mountedBy":
                s_s[2] = 0
                s_s[0] = 1
        return s_s

    def get_b_e(self, preds):
        b_e = [0, 0, 0, 0, 0]
        for pred in preds:
            pred = pred.split()
            if pred[0] == "batteryRemain":
                if int(pred[2]) > 80:
                    b_e[4] = 1
                elif int(pred[2]) > 60:
                    b_e[3] = 1
                elif int(pred[2]) > 40:
                    b_e[2] = 1
                elif int(pred[2]) > 20:
                    b_e[1] = 1
                else:
                    b_e[0] = 1
        return b_e

    def get_t_e(self, preds):
        t_e = [0, 0, 0, 0, 0]
        for pred in preds:
            pred = pred.split()
            if pred[0] == "distance":
                if float(pred[3]) > 35:
                    t_e[0] = 1
                elif float(pred[3]) > 28:
                    t_e[1] = 1
                elif float(pred[3]) > 20:
                    t_e[2] = 1
                elif float(pred[3]) > 13:
                    t_e[3] = 1
                else:
                    t_e[4] = 1
        return t_e

    def policyParmeter(self, datas):
        

        predicate = datas #self.get_randState()
        bs = self.get_b_s(predicate)
        cs = self.get_c_s(predicate)
        ss = self.get_s_s(predicate)
        be = self.get_b_e(predicate)
        te = self.get_t_e(predicate)
        pp = [cs, bs, ss, be, te]
        pp = np.array(pp)
        
        return pp[0], pp[1], pp[2], pp[3], pp[4]

    #goal checking / not implement
    def is_goal(self):

        if self.goal_type=="cloud":
            #print(self.state)
            if self.state[0, 0] == 1 and self.state[1, 2] == 1 and self.state[2, 1] == 1:
                return True
            else:
                return False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_rwd(self, action):
        action_index = int(action / 5)
        action_value = int(action % 5)
        reward = 0
        safety_c, safety_b, safety_l = self.state[0], self.state[1], self.state[2]

        safety_c = safety_c.tolist()
        safety_b = safety_b.tolist()
        safety_l = safety_l.tolist()

        c =safety_c.index(1)
        b =safety_b.index(1)
        l =safety_l.index(1)
        
        v = (c + b + l) / 3  

        judge = abs(action_value - v)
        scale = 1
        if judge > -0.5 and judge < 0.5:
           reward = 3
        else:
           reward = -judge

        return reward


    #step func
    def step(self, action):
        return self.state, self.get_rwd(action), True, {}

    def reset(self,random=False,allow_goal=False):
        self.episode=0
        if not random:
            self.state = self.get_pred()
            return self.state
        
        else:
            self.state = self.get_random_state()
            return self.get_obs()
            print('not implemented')
            exit(0)


        self.state = self.get_random_state()
        
        if not allow_goal:
            while self.is_goal():
                self.state = self.get_random_state()
        
        return self.self.get_pred()
