import argparse
import csv
import os

import numpy as np
from collections import deque, defaultdict

from addition_rnn_model import AdditionRNNModel, AdditionRNNEnvironment
from addition_rnn_curriculum import gen_curriculum_baseline, gen_curriculum_naive, gen_curriculum_mixed, gen_curriculum_combined
from tensorboard_utils import create_summary_writer, add_summary

class EXP3(object):
    def __init__(self, number_of_arms=9, gamma=0.2):
        self._number_of_arms = number_of_arms
        self.name = 'exp3 Gamma: ' + str(gamma)
        self.action_values = np.zeros((2,number_of_arms))
        self.gamma = gamma
        self.weights = np.ones((1,number_of_arms))
        self.time = 0.
        self.previous_action = None
        self.reset()
  
    def __call__(self, reward):
        if self.previous_action != None:
            xhat = np.zeros((1, self._number_of_arms))
            xhat[0,self.previous_action] = reward/self.action_values[0,self.previous_action]
            self.weights = self.weights*np.exp(self.gamma*xhat/self._number_of_arms)
            self.action_values[1,self.previous_action] += 1.
        self.action_values[0,:] = (1-self.gamma)*(self.weights)/(np.sum(self.weights)) + self.gamma/self._number_of_arms
        action = np.random.choice(self._number_of_arms, p=self.action_values[0,:])
        self.time += 1.
        unvisited = np.where(self.action_values[1,:] == 0.)
        self.previous_action = unvisited[0][0] if unvisited[0].size > 0 else action
        return self.previous_action
  
    def getProbs(self):
        return self.action_values[0,:]
  
    def reset(self):
        self.action_values = np.zeros((2, self._number_of_arms))
        self.weights = np.ones((1, self._number_of_arms))
        self.time = 0
        self.previous_action = None
        return

class UCB(object):
    def __init__(self, number_of_arms=9):
        self._number_of_arms = number_of_arms
        self.name = 'ucb'
        self.action_values = np.zeros((2,number_of_arms))
        self.time = 0.
        self.reset()

    def __call__(self, reward):
        if self.previous_action != None:
            self.action_values[0,self.previous_action] += reward
            self.action_values[1,self.previous_action] += 1.
        self.time += 1.
        unvisited = np.where(self.action_values[1,:] == 0.)
        self.previous_action = unvisited[0][0] if unvisited[0].size > 0 else np.argmax(np.sqrt(np.log(self.time)/self.action_values[1,:]) + np.divide(self.action_values[0,:],self.action_values[1,:]))
        return self.previous_action

    def getProbs(self):
        # Not really probs
        return self.action_values[0,:]

    def reset(self):
        self.action_values = np.zeros((2,number_of_arms))
        self.time = 0
        self.previous_action = None
        return

class REINFORCE(object):
    def __init__(self, number_of_arms=9, step_size=0.1, baseline=True):
        self._number_of_arms = number_of_arms
        self._lr = step_size
        self.name = 'reinforce, baseline: {}'.format(baseline)
        self._baseline = baseline
        self.action_values = np.zeros((2,number_of_arms))
        self.action_preferences = np.zeros((1,number_of_arms))
        self.total_reward = 0;
        self.number_rewards = 0.
        self.previous_action = None
        self.reset()
  
    def __call__(self, reward):
        if self.previous_action != None:
            self.number_rewards += 1.
            self.total_reward += reward
            self.action_values[0,self.previous_action] += reward
            self.action_values[1,self.previous_action] += 1.
            self.updatePreferences(self.previous_action, reward)
        self.previous_action = np.random.choice(np.arange(0,self._number_of_arms),p=self.softmax())
        return self.previous_action
    
    def reset(self):
        self.action_values = np.zeros((2,self._number_of_arms))
        self.action_preferences = np.zeros((1,self._number_of_arms))
        self.number_rewards = 0.
        self.total_reward = 0.
        self.previous_action = None
  
    def updatePreferences(self, previous_action, reward):
        if not self._baseline: 
            self.action_preferences[0,previous_action]+=self._lr*reward*(1-self.softmax()[previous_action])
            for i in range(0,self._number_of_arms):
                if i != previous_action:
                    self.action_preferences[0,i]-=self._lr*reward*self.softmax()[i]
        else:
            self.action_preferences[0,previous_action]+=self._lr*(reward - self.total_reward/self.number_rewards)*(1-self.softmax()[previous_action])
            for i in range(0,self._number_of_arms):
                if i != previous_action:
                    self.action_preferences[0,i]-=self._lr*(reward - self.total_reward/self.number_rewards)*self.softmax()[i]
    
    def softmax(self):
        q = np.sum(np.exp(self.action_preferences),axis=1)
        t = np.exp(self.action_preferences)/q
        return t.flatten()
  
    def getProbs(self):
        return self.softmax()


def estimate_slope(x, y):
    assert len(x) == len(y)
    A = np.vstack([x, np.ones(len(x))]).T
    c, _ = np.linalg.lstsq(A, y)[0]
    return c


class CurriculumTeacher:
    def __init__(self, env, curriculum, writer=None):
        """
        'curriculum' e.g. arrays defined in addition_rnn_curriculum.DIGITS_DIST_EXPERIMENTS
        """
        self.env = env
        self.curriculum = curriculum
        self.writer = writer

    def teach(self, num_timesteps=2000):
        curriculum_step = 0
        for t in range(num_timesteps):
            p = self.curriculum[curriculum_step]
            print(p)
            r, train_done, val_done = self.env.step(p)
            if train_done and curriculum_step < len(self.curriculum)-1:
                curriculum_step = curriculum_step + 1
            if val_done:
                return self.env.model.epochs

            if self.writer:
                for i in range(self.env.num_actions):
                    add_summary(self.writer, "probabilities/task_%d" % (i + 1), p[i], self.env.model.epochs)

        return self.env.model.epochs


class WindowedSlopeBanditTeacher:
    def __init__(self, env, policy, window_size=10, abs=False, writer=None):
        self.env = env
        self.policy = policy
        self.window_size = window_size
        self.abs = abs
        self.scores = [deque(maxlen=window_size) for _ in range(env.num_actions)]
        self.timesteps = [deque(maxlen=window_size) for _ in range(env.num_actions)]
        self.writer = writer

    def teach(self, num_timesteps=2000):
        chosen_action = 0
        print('Initial Chosen Action:', chosen_action)
        for t in range(num_timesteps):
            slopes = [estimate_slope(timesteps, scores) if len(scores) > 1 else 1 for timesteps, scores in zip(self.timesteps, self.scores)]
                
            if self.env.signal == 'SPG':
                 reward = np.abs(slopes[chosen_action]) if self.abs else slopes[chosen_action]
            elif self.env.signal == 'MPG':
                 reward = np.mean(np.abs(slopes) if self.abs else slopes)
                 
            #p = self.policy(np.abs(slopes) if self.abs else slopes)
            p = self.policy(reward)
            temp = np.zeros(self.env.num_actions)
            temp[p] = 1.
            p = temp.copy()
            r, train_done, val_done = self.env.step(p)
            if val_done:
                return self.env.model.epochs
            for a, s in enumerate(r):
                if not np.isnan(s):
                    self.scores[a].append(s)
                    self.timesteps[a].append(t)

            if self.writer:
                for i in range(self.env.num_actions):
                    add_summary(self.writer, "slopes/task_%d" % (i + 1), slopes[i], self.env.model.epochs)
                    add_summary(self.writer, "probabilities/task_%d" % (i + 1), p[i], self.env.model.epochs)

        return self.env.model.epochs




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--teacher', choices=['curriculum', 'naive', 'online', 'window', 'sampling'], default='sampling')
    parser.add_argument('--curriculum', choices=['uniform', 'naive', 'mixed', 'combined'], default='combined')
    parser.add_argument('--policy', choices=['reinforce', 'exp3', 'ucb3'], default='thompson')
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.0004)
    parser.add_argument('--bandit_lr', type=float, default=0.1)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--abs', action='store_true', default=False)
    parser.add_argument('--no_abs', action='store_false', dest='abs')
    parser.add_argument('--max_timesteps', type=int, default=20000)
    parser.add_argument('--max_digits', type=int, default=9)
    parser.add_argument('--invert', action='store_true', default=True)
    parser.add_argument('--no_invert', action='store_false', dest='invert')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--train_size', type=int, default=40960)
    parser.add_argument('--val_size', type=int, default=4096)
    parser.add_argument('--optimizer_lr', type=float, default=0.001)
    parser.add_argument('--clipnorm', type=float, default=2)
    parser.add_argument('--logdir', default='addition')
    parser.add_argument('run_id')
    parser.add_argument('--csv_file')
    args = parser.parse_args()

    logdir = os.path.join(args.logdir, args.run_id)
    writer = create_summary_writer(logdir)

    model = AdditionRNNModel(args.max_digits, args.hidden_size, args.batch_size, args.invert, args.optimizer_lr, args.clipnorm)

    val_dist = gen_curriculum_baseline(args.max_digits)[-1]
    env = AdditionRNNEnvironment(model, 'MPG', args.train_size, args.val_size, val_dist, writer)

    if args.teacher != 'curriculum':
        if args.policy == 'reinforce':
            policy = REINFORCE()
        elif args.policy == 'ucb':
            policy = UCB()
        elif args.policy == 'exp3':
            policy = EXP3()
        else:
            assert False

    if args.teacher == 'naive':
        teacher = NaiveSlopeBanditTeacher(env, policy, args.bandit_lr, args.window_size, args.abs, writer)
    elif args.teacher == 'online':
        teacher = OnlineSlopeBanditTeacher(env, policy, args.bandit_lr, args.abs, writer)
    elif args.teacher == 'window':
        teacher = WindowedSlopeBanditTeacher(env, policy, args.window_size, args.abs, writer)
    elif args.teacher == 'sampling':
        teacher = SamplingTeacher(env, policy, args.window_size, args.abs, writer)
    elif args.teacher == 'curriculum':
        if args.curriculum == 'uniform':
            curriculum = gen_curriculum_baseline(args.max_digits)
        elif args.curriculum == 'naive':
            curriculum = gen_curriculum_naive(args.max_digits)
        elif args.curriculum == 'mixed':
            curriculum = gen_curriculum_mixed(args.max_digits)
        elif args.curriculum == 'combined':
            curriculum = gen_curriculum_combined(args.max_digits)
        else:
            assert False

        teacher = CurriculumTeacher(env, curriculum, writer)
    else:
        assert False

    epochs = teacher.teach(args.max_timesteps)
    print("Finished after", epochs, "epochs.")

    if args.csv_file:
        data = vars(args)
        data['epochs'] = epochs
        header = sorted(data.keys())

        # write the CSV file one directory above the experiment directory
        csv_file = os.path.join(args.logdir, args.csv_file)
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a') as file:
            writer = csv.DictWriter(file, delimiter=',', fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
