import sys
sys.path.append("/home/kist/pythonProject/Python-mcArbiFramework")

import numpy as np
import tensorflow as tf

from CloudEnv import *
from cloud_ilp_def import *
from cloud_ilp_def_eff import *
import argparse
from threading import Condition, Thread
from arbi_agent.agent.arbi_agent import ArbiAgent
from arbi_agent.ltm.data_source import DataSource
from arbi_agent.agent import arbi_agent_executor
from arbi_agent.model import generalized_list_factory as GLFactory
import time

params = dotdict({})
params.ILP_VALUE=False
params.HARD_CHOICE=False
params.DBL_SOFTMAX=False
params.REMOVE_REP=False
params.RESET_RANDOM=False
params.MAX_EPISODES=20000
params.MAX_STEPS=100
params.EPS_TH=0
params.MAX_MEM_SIZE=100
params.NUM_BOX=4
params.LR_ACTOR=.02
params.LR_VALUE=.02
params.DISCOUNT_GAMMA=1
params.REWARD=1
params.PENALTY=-.01
params.ATTEN=1
params.SOFTMAX_COEFF= lambda x: x
IMG_SIZE = 64


MODELPATH = './model/safety_model'
LOGPATH = './model/safety_model/log'
LOAD = False
TESTMODE = False


env = CloudWorldEnvImage(goal_type='cloud', reward=params.REWARD,penalty=params.PENALTY,error_penalty=params.PENALTY)

print(params)

ilp_mdl = ILP_MODEL(num_box=params.NUM_BOX)
ilp_mdl_eff = ILP_MODEL_EFF(num_box=params.NUM_BOX)
ilp_mdl.mdl.print_vars()

class Memory(object):
    def __init__(self):
        self.ep_obs, self.ep_obs1, self.ep_obs2,  self.ep_act, self.ep_rwd = [], [], [], [], []

    def store_transition(self, obs0, obs1, obs2, act, rwd):
        self.ep_obs.append(obs0)
        self.ep_obs1.append(obs1)
        self.ep_obs2.append(obs2)
        self.ep_act.append(act)
        self.ep_rwd.append(float(rwd))

    def covert_to_array(self):
        array_obs = np.stack(self.ep_obs, 0)
        array_obs1 = np.stack(self.ep_obs1, 0)
        array_obs2 = np.stack(self.ep_obs2, 0)
        array_act = np.array(self.ep_act)
        array_rwd = np.array(self.ep_rwd)
        return array_obs, array_obs1, array_obs2, array_act, array_rwd

    def reset(self):
        self.ep_obs,self.ep_obs1, self.ep_obs2, self.ep_act, self.ep_rwd = [], [], [], [], []

    def limit_size(self):
        if len(self.ep_act)>params.MAX_MEM_SIZE:
            self.ep_act = self.ep_act[-params.MAX_MEM_SIZE:]
            self.ep_obs = self.ep_obs[-params.MAX_MEM_SIZE:]
            self.ep_obs1 = self.ep_obs1[-params.MAX_MEM_SIZE:]
            self.ep_obs2 = self.ep_obs2[-params.MAX_MEM_SIZE:]
            self.ep_rwd = self.ep_rwd[-params.MAX_MEM_SIZE:]

    def cleanup(self):
        for i in range( len(self.ep_obs)-2):
            for j in range(i+1,len(self.ep_obs)-1):
                if np.all(self.ep_obs[i]==self.ep_obs[j]):
                    self.ep_obs=self.ep_obs[:i]+self.ep_obs[j:]
                    self.ep_obs1=self.ep_obs1[:i]+self.ep_obs[j:]
                    self.ep_obs2=self.ep_obs2[:i]+self.ep_obs[j:]
                    self.ep_act=self.ep_act[:i]+self.ep_act[j:]
                    self.ep_rwd=self.ep_rwd[:i]+self.ep_rwd[j:]
                    i=j
                    break

class ActorCritic:
    def __init__(self, lr_actor, lr_value, gamma, ilp, name):

        self.lr_actor = lr_actor
        self.lr_value = lr_value
        self.gamma = gamma
        self.OBS = tf.placeholder(tf.float32, [None, IMG_SIZE,IMG_SIZE,3], name="observation")
        self.INPUT = tf.placeholder(tf.float32, [None,5], name='input')
        self.INPUT1 = tf.placeholder(tf.float32, [None,5], name='input1')
        self.INPUT2 = tf.placeholder(tf.float32, [None,5], name='input2')
        self.INPUT3 = tf.placeholder(tf.float32, [None,5], name='input3')
        self.INPUT4 = tf.placeholder(tf.float32, [None,5], name='input4')
        self.ACT = tf.placeholder(tf.int32, [None], name="action")
        self.Q_VAL = tf.placeholder(tf.float32, [None, 1], name="q_value")
        self.tf_suc = tf.placeholder(shape=[None], dtype=tf.float32)
        self.tf_reward = tf.placeholder(shape=[None], dtype=tf.float32)
        self.nstep =0
        self.name= name

        self.memory = Memory()
        # run ILP to generate actions
        if name == "eff":
            with tf.variable_scope("ILP", reuse=False):
                # print("INPUT", self.INPUT, self.INPUT1, self.INPUT2, self.INPUT3, self.INPUT4)
                self.action, self.xo = ilp.run(self.INPUT, self.INPUT1)
        else:
            with tf.variable_scope("ILP", reuse=False):
                # print("INPUT", self.INPUT, self.INPUT1, self.INPUT2, self.INPUT3, self.INPUT4)
                self.action, self.xo = ilp.run(self.INPUT, self.INPUT1, self.INPUT2)

        self.probs = self.action

        self.act = tf.nn.softmax(self.probs*12)
        self.merge = tf.summary.merge_all()
        # loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.probs*12, labels=self.ACT)
        self.advantage = self.Q_VAL
        self.actor_loss = tf.reduce_mean(cross_entropy * self.advantage)
        self.loss_scalar = tf.compat.v1.summary.scalar("loss", self.actor_loss)

        self.actor_train_op = tf.train.AdamOptimizer(self.lr_actor).minimize(self.actor_loss)
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.merged_summary = tf.compat.v1.summary.merge_all()
        self.saver = tf.train.Saver()
        if LOAD:
            self.load(self.sess, MODELPATH)
        self.writer = tf.compat.v1.summary.FileWriter(LOGPATH, self.sess.graph)
        #with tf.Session() as sess:
        #    self.writer = tf.summary.FileWriter('./log/', sess.graph)


    def save(self, sess, path):
        self.saver.save(sess, path + "/parameters.ckpt")
        # if self.critic and isinstance(self.critic, TableCritic):
        #    self.critic.save(path + "/critic.pl")

    def load(self, sess, path):
        self.saver.restore(sess, path + "/parameters.ckpt")

    def step(self, x, x1, x2, episode, pType):
        x = x[np.newaxis, :]
        x1 = x1[np.newaxis, :]
        if pType == "safety":
            x2 = x2[np.newaxis, :]
            prob_weights = self.sess.run(self.act, feed_dict={self.INPUT: x, self.INPUT1:x1, self.INPUT2:x2})
            p = prob_weights.ravel()
            value = self.sess.run(self.xo, feed_dict={self.INPUT: x, self.INPUT1:x1, self.INPUT2:x2})

        elif pType == "eff":
            prob_weights = self.sess.run(self.act, feed_dict={self.INPUT: x, self.INPUT1: x1})
            p = prob_weights.ravel()
            value = self.sess.run(self.xo, feed_dict={self.INPUT: x, self.INPUT1: x1})

        action = np.random.choice(range(prob_weights.shape[1]), p=p)
        value = 0
        return action, value

    def learn(self, last_value, done, reset_mem=True, step=0):
        x, x1, x2, act, rwd = self.memory.covert_to_array()
        q_value = self.compute_q_value(last_value, done, rwd)
        self.sess.run(self.actor_train_op, {self.INPUT: x, self.INPUT1:x1, self.INPUT2:x2, self.ACT: act, self.Q_VAL: q_value})
        summary = self.sess.run(self.merged_summary,{self.INPUT: x, self.INPUT1:x1, self.INPUT2:x2, self.ACT: act, self.Q_VAL: q_value})
        self.writer.add_summary(summary, step)
        self.save(self.sess, MODELPATH)
        self.memory.reset()

    def compute_q_value(self, last_value, done, rwd):
        q_value = np.zeros_like(rwd)
        value = 0 if done else last_value
        for t in reversed(range(0, len(rwd))):
            value = value * self.gamma + rwd[t]
            q_value[t] = value
        return q_value[:, np.newaxis]


sAgent = ActorCritic(lr_actor=params.LR_ACTOR, lr_value=params.LR_VALUE, gamma=params.DISCOUNT_GAMMA, ilp=ilp_mdl, name="safey")
eAgent = ActorCritic(lr_actor=params.LR_ACTOR, lr_value=params.LR_VALUE, gamma=params.DISCOUNT_GAMMA, ilp=ilp_mdl_eff, name="eff")


############### for ARBI Agent##################
#broker_url = "tcp://127.0.0.1:61319"
broker_url = "tcp://203.249.22.57:61319"
JMS_Broker_URL = "tcp://127.0.0.1:61319"
agent_name = "agent://www.arbi.com/TaskPolicyLearner"
class TaskPolicyLeanerAgent(ArbiAgent):
    def __init__(self):
        super().__init__()
        self.lock = Condition()
        self.TPL_name = "agent://www.arbi.com/TaskPolicyLearner"
        self.TaskManager_name= "agent://www.arbi.com/TaskManager"
        self.CM_name = "agent://www.arbi.com/ContextManager"
        self.index = 0
        self.processing = True
        self.robots = ["http://www.arbi.com/ontologies/arbi.owl#AMR_LIFT1", "http://www.arbi.com/ontologies/arbi.owl#AMR_LIFT2"]
        self.contexts = ["robotAtVertex $Robot $Vertex", "cargoOnVertex $Cargo $Vertex", "batteryRemain $Robot $Battery", "distance $Robot $Cargo $Distance"]

    def on_start(self):
        print('\n** TaskPolicy Learner Agent Start ***!!\n')


    def convert_pred(self, gl):
        x= ""
        gl = gl.get_expression(0)
        glnew = gl.as_generalized_list()
        n = glnew.get_expression_size()

        for i in range(0,n+1):
            if i == 0:
                x = glnew.get_name()
            else:
                x = x+" " + glnew.get_expression(i-1).as_value().string_value()
            print(x)
        return x

    def get_predicate(self, robot, cargo):
        pred = []#init
        time.sleep(1)
        for context in self.contexts:
            context = context.split()
            msg = "(context (" + context[0]
            for i in range(len(context)-1):
                if context[i+1] == "$Cargo":
                    msg= msg + " \"{0}\"".format(cargo)
                elif context[i+1] == "$Robot":
                    msg = msg + " \"{0}\"".format(robot)
                else:
                    print("wrong!")
            msg = msg +" "+context[-1] + "))"
            queryResult = self.query(self.CM_name, msg)
            print(queryResult)
            res = GLFactory.new_gl_from_gl_string(queryResult)
            context = self.convert_pred(res)
            pred.append(context)
            time.sleep(1)

        return pred

    def on_request(self, sender: str, request: str) -> str:
        # (policy $goal_id $goal_name $cargo_id)
        # (policy "goal001" "palletTransported" "cargo01")
        print("Sender : " + sender)
        print("Request : "+ request)
        request= GLFactory.new_gl_from_gl_string(request)
        taskName =request.get_expression(1).as_value().string_value()
        cargo = request.get_expression(2).as_value().string_value()
        goal = request.get_expression(0)


        time.sleep(1)
        if sender == self.TaskManager_name:
            for r in self.robots:
                pred = self.get_predicate(r, cargo)
                s_c, s_b, s_l, e_b, e_t = env.policyParmeter(pred)
                print(s_c, s_b, s_l)
                print(e_b, e_t)
                act, _ = sAgent.step(s_c, s_b, s_l, 0, "safety")
                act2, _ = eAgent.step(e_t, e_b, _, 0, "eff")
                safety = act-2
                efficiency = act2-2
                print(r+" Safety Grade     : ", safety)
                print(r+" Efficiency Grade : ", efficiency)

                msg = "(policy (safety \"{0}\" \"{1}\" \"{2}\" \"{3}\"))".format(r, goal, taskName, safety)
                self.send(self.TaskManager_name, msg)
                time.sleep(1)
                msg = "(policy (efficiency \"{0}\" \"{1}\" \"{2}\" \"{3}\"))".format(r, goal, taskName, efficiency)
                self.send(self.TaskManager_name, msg)
                time.sleep(1)

        return "ok"

    def on_data(self, sender: str, data: str):
        print("Sender: " + sender)
        print("Data: " +data)
        data = GLFactory.new_gl_from_gl_string(data)
        robot = data.get_expression(0).as_value().string_value()
        robot = robot[-9:]
        if robot == "AMR_LIFT1":
            self.robots.remove("http://www.arbi.com/ontologies/arbi.owl#AMR_LIFT1")
        elif robot == "AMR_LIFT2":
            self.robots.remove("http://www.arbi.com/ontologies/arbi.owl#AMR_LIFT2")


if __name__ == '__main__':
    arbiAgent = TaskPolicyLeanerAgent()
    arbi_agent_executor.execute(broker_url=broker_url, agent_name="agent://www.arbi.com/TaskPolicyLearner",
                                agent=arbiAgent, broker_type=2)  # same role with agent.initialize
    while True:
            time.sleep(1)

    agent.close()
