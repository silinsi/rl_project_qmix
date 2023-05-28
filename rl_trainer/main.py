import argparse
import datetime
import numpy as np
from tensorboardX import SummaryWriter

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from algo.ddpg import DDPG
from common import *
from log_path import *
from env.chooseenv import make
from replay_buffer import ReplayBuffer
from qmix import QMIX
from normalization import Normalization



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Runner_QMIX:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = args.game_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = make(self.env_name, conf=None)

        #self.env_info = self.env.get_env_info()
        self.args.N = 6  # The number of agents
        self.ctrl_agent_index = [0, 1, 2]
        self.args.obs_dim = 26  # The dimensions of an agent's observation space
        self.args.state_dim = 6  # The dimensions of global state space
        self.args.action_dim = self.env.get_action_dim() # The dimensions of an agent's action space
        #self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        print("number of max_train_steps={}".format(self.args.max_train_steps))
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        self.agent_n = QMIX(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='./runs/{}/{}_env_{}_number_{}_seed_{}'.format(self.args.algorithm, self.args.algorithm, self.env_name, self.number, self.seed))

        self.epsilon = self.args.epsilon  # Initialize the epsilon
        self.win_rates = []  # Record the win rates
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)

        self.obs_dim=26;
        self.height=self.env.board_height;
        self.width=self.env.board_width;

    def get_dir(self,):#get the last model direction
        dir="model\env"
        rlist=os.listdir(dir)
        a=[]
        b=[]
        for i in range(len(rlist)):
            a.append("")
            for j in rlist[i]:
                if j<"0" or j>"9":
                    if len(a[i])>0:
                        if a[i][-1]!="s":
                            a[i]+="s"
                else:
                    a[i]+=j
            b.append(a[i].split("s")[:-1])
            a[i]=[a[i].split("s")[2]]
            for j in range(len(a[i])):
                a[i][j]=int(a[i][j])
            k=np.argmax(a)
        return dir+"\\"+rlist[k],int(b[k][0]),int(b[k][2])


    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        
        #load model
        if self.args.load_model:
            dir,load_eva,load_num=self.get_dir()
            torch.load( dir)
            evaluate_num=load_eva
            self.total_steps=load_num*1000
            self.args.max_train_steps+=self.total_steps
            print("model loaded "+ dir)

        
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                evaluate_num += 1
                self.evaluate_policy(evaluate_num)  # Evaluate the policy every 'evaluate_freq' steps
                

            _, _, episode_steps = self.run_episode_snake(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.current_size >= self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
        evaluate_num+=1
        self.evaluate_policy(evaluate_num)
        print("end");
        #self.env.close()

    def evaluate_policy(self, evaluate_num):
        win_times = 0
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _ = self.run_episode_snake(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.win_rates.append(win_rate)
        score=np.sum(evaluate_reward[0:3])-np.sum(evaluate_reward[3:6])
        print("total_steps:{} \t score:{} \t evaluate_reward:{}".format(self.total_steps, score, evaluate_reward))
        self.writer.add_scalar('win_rate_{}'.format(self.env_name), win_rate, global_step=self.total_steps)
        self.agent_n.save_model('env','qmix',evaluate_num,self.seed,self.total_steps)
        
        #print("save")
        # Save the win rates
        #np.save('./data_train/{}_env_{}_number_{}_seed_{}.npy'.format(self.args.algorithm, self.env_name, self.number, self.seed), np.array(self.win_rates))

    def run_episode_snake(self, evaluate=False):
        win_tag = False
        episode_reward = np.zeros(6)
        state=self.env.reset();
        state_to_training = state[0];
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.eval_Q_net.rnn_hidden = None
        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))  # Last actions of N agents(one-hot)
        for episode_step in range(self.args.episode_length):
            obs_n = get_observations(state_to_training, self.ctrl_agent_index, self.obs_dim, self.height, self.width)
            #obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
            #s = self.env.get_state()  # s.shape=(state_dim,)
            #avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            avail_a_n=np.ones((6,self.env.get_action_dim()));
            epsilon = 0 if evaluate else self.epsilon
            a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)
            #print(type(a_n_[0]));

            #print(a_n);
            #break;
            
            last_onehot_a_n = np.eye(self.args.action_dim)[a_n]  # Convert actions to one-hot vectors
            s, r, done, _, info = self.env.step(self.env.encode(a_n))  # Take a step
            r=np.array(r);
            #win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)
                # Decay the epsilon
                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min

            if done:
                break

        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            #obs_n = self.env.get_obs()
            '''obs_n = get_observations(state_to_training, self.ctrl_agent_index, self.obs_dim, self.height, self.width)
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            self.replay_buffer.store_last_step(episode_step + 1, obs_n, s, avail_a_n)'''
            pass
            

        return win_tag, episode_reward, episode_step + 1



def main(args):
    runner=Runner_QMIX(args,None,number=1, seed=0);
    runner.run();

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="snakes_3v3", type=str)
    parser.add_argument('--algo', default="ddpg", type=str, help="bicnet/ddpg")
    #parser.add_argument('--max_episodes', default=50000, type=int)
    parser.add_argument('--episode_length', default=200, type=int)
    parser.add_argument('--episode_limit', default=200, type=int)
    parser.add_argument('--output_activation', default="softmax", type=str, help="tanh/softmax")

    #parser.add_argument('--buffer_size', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=1, type=int)
    #parser.add_argument('--a_lr', default=0.0001, type=float)
    #parser.add_argument('--c_lr', default=0.0001, type=float)
    #parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epsilon', default=0.5, type=float)
    #parser.add_argument('--epsilon_speed', default=0.99998, type=float)

    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
    #parser.add_argument("--load_model_run", default=2, type=int)
    #parser.add_argument("--load_model_run_episode", default=4000, type=int)
    #parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")

    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=32, help="Evaluate times")
    #parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")

    parser.add_argument("--algorithm", type=str, default="QMIX", help="QMIX or VDN")
    #parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_decay_steps", type=float, default=50000, help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--buffer_size", type=int, default=5000, help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    #parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--qmix_hidden_dim", type=int, default=32, help="The dimension of the hidden layer of the QMIX network")
    parser.add_argument("--hyper_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the hyper-network")
    parser.add_argument("--hyper_layers_num", type=int, default=1, help="The number of layers of hyper-network")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="use lr decay")
    parser.add_argument("--use_RMS", type=bool, default=False, help="Whether to use RMS,if False, we will use Adam")
    parser.add_argument("--add_last_action", type=bool, default=True, help="Whether to add last actions into the observation")
    parser.add_argument("--add_agent_id", type=bool, default=True, help="Whether to add agent id into the observation")
    parser.add_argument("--use_double_q", type=bool, default=True, help="Whether to use double q-learning")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Whether to use reward normalization")
    parser.add_argument("--use_hard_update", type=bool, default=True, help="Whether to use hard update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network")
    #parser.add_argument("--tau", type=int, default=0.005, help="If use soft update")

    #parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    #parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
    #parser.add_argument("--epsilon_decay_steps", type=float, default=50000, help="How many steps before the epsilon decays to the minimum")

    args = parser.parse_args()

    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps

    main(args)
