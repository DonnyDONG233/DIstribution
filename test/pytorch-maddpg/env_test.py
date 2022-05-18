# from sympy.codegen.tests.test_fnodes import np
import numpy as np
import random
import math

def BW_to_data_rate(BW):
    SNR =20
    return BW * np.log2(1+SNR)

#下面的信息是为了随机产生Task实例
class GlobalVar:
    computation_resource_rand_min = 0
    computation_resource_rand_max = 1
    data_size_rand_min = 1e6
    data_size_rand_max = 1e7
    max_delay = 5e-3
    slot = 5e-4

class Agent:
    def __init__(self, id):
        self.id = id

class Agent_mec:
    def __init__(self, id, com_re, bw_re):
        """每次对agent进行观测，都会在task_list中产生任务实例，然后将其放到queued_task列表末尾"""
        self.com_resource = com_re #基站的计算能力
        self.id = id
        self.bw_resource = bw_re #基站的无线带宽
        self.com_left = self.com_resource #基站当前剩余的计算资源
        self.bw_left = self.bw_resource  #基站当前剩余的带宽资源
        self.task_list = [] #临时用于记录新到来的请求，当任务被加载到任务队列之后，将其清空
        self.queued_task = []
        self.processing_task_list = []
        self.timesteps = 0
        self._timesteps_limit = 1000
        self.generate_task_for_observation()

    def timestep_update(self):
        self.timesteps += 1

    def __repr__(self):
        return "agent id: "+str(self.id) + " with Computing resource: "+str(self.com_resource) + "and uplink BW resource:"+str(self.bw_resource)
        #return "AAAAAa"

    def reset(self):
        self.timesteps = 0
        self.queued_task = []
        self.generate_task_for_observation()
        self.processing_task_list = []
        self.com_left = self.com_resource
        self.bw_left = self.bw_resource
        current_process_task = self.queued_task[0]
        return [self.com_left, self.bw_left, current_process_task.datasize, current_process_task.com_size]



    def generate_task_for_observation(self):
        """这里简单起见，每次只生成一个任务
           将其添加在任务队列尾部
           设置task的id为{“current_timestep”:value,agent_id:task_num}
        """
        compusize = random.randint(GlobalVar.computation_resource_rand_min,GlobalVar.computation_resource_rand_max)
        datasize = random.randint(int(GlobalVar.data_size_rand_min),int(GlobalVar.data_size_rand_max))
        task = Task(compusize,datasize,GlobalVar.max_delay)
        #设置task_id
        task_id = {'current_timestep': self.timesteps, self.id: 1}
        task.set_id(task_id)
        #将生成的任务放进self.task_list
        self.task_list.append(task)
        # 将self.task_list中的任务全部放进self.queued_task列表后面，并清空self.task_list
        self.queued_task.extend(self.task_list)
        self.task_list=[]


    # def update_resource(self):
    #     for p_task in self.processing_task_list:
    #         if p_task.bw


    # def agent_reset(self):
    #     self.timesteps = 0
    #     return None
    def allocate_resource(self,task,BW,CPU):
        task.com_occupied = CPU
        task.bw_occupied = BW
        task.trans_time_slot = math.ceil(task.datasize/BW_to_data_rate(BW)/GlobalVar.slot)
        task.com_time_slot = math.ceil(task.com_size/CPU/GlobalVar.slot)

    def resource_release(self,task):
        if task.bw_released != True:
            task.trans_time_slot -= 1
            if task.trans_time_slot == 0:
                self.bw_resource += task.bw_occupied
                task.bw_released = True
        if task.bw_released == True and task.com_released != True:
            task.com_time_slot -= 1
            if task.com_time_slot == 0:
                self.com_resource += task.com_occupied
                task.com_released = True
        if task.bw_released == True and task.com_released == True:
            self.processing_task_list.remove(task)


    def get_obs(self,action_dict):
        """在执行本函数时，生成新的任务放入queued_taks列表后面，同时观测queued_task列表头部的任务"""
        self.generate_task_for_observation()
        agent_allocate_BW = action_dict['BW']
        agent_allocate_CPU = action_dict['CPU']
        self.com_left -= agent_allocate_CPU
        self.bw_left -= agent_allocate_BW
        current_process_task =self.queued_task[0]
        self.allocate_resource(current_process_task,agent_allocate_BW,agent_allocate_CPU)
        #这一句可以放在判断reward函数的代码里边self.processing_task_list.append(current_process_task)
        #processed_task = self.queued_task.pop(0)
        return [self.com_left, self.bw_left, current_process_task.datasize, current_process_task.com_size]


class Task:
    def __init__(self, com_size, datasize, max_delay):
        self.com_size = com_size
        self.datasize = datasize
        self.com_occupied = 0
        self.bw_occupied = 0
        self.com_released = None
        self.bw_released = None
        self.trans_time_slot = None
        self.com_time_slot = None
        self.max_delay = max_delay
        self.ID = None
    def set_id(self,ID):
        self.ID = ID

    # def resource_allocate(self,BW,CPU):
    #     self.com_occupied = CPU
    #     self.bw_occupied = BW
    #     self.trans_time_slot = math.ceil(self.datasize/BW_to_data_rate(BW)/GlobalVar.slot)
    #     self.com_time_slot = math.ceil(self.com_size/CPU/GlobalVar.slot)





    def reward(self):
        if self.trans_time_slot + self.com_time_slot <= math.ceil(self.max_delay/GlobalVar.slot)+1:
            return 1
        else:
            return 0
    # def process(self):
    #     if self.trans_time < 0:
    #         self.com_time -= 1
    #     else:
    #         self.trans_time -= 1
    #     if self.com_time < 0:
    #         return True



class Env_test:

    def __init__(self, agent_num):
        self.agent_num = agent_num
        self.state_dim = 2
        self.action_dim = 2
        self.agents = []
        self.reward_mech = 'global'
        for i in range(self.agent_num):
            self.agents.append(Agent(i))

        self.episode_length = 20000

        self.states = [random.randint(10,100) for i in range(self.state_dim)]
        self.step_num = 0
        self._timesteps = 0





    @property
    def timestep_limit(self):
        return 1000


    def get_param_values(self):
        return self.__dict__

    def reset(self):
        self.states = [random.randint(10, 100) for i in range(self.state_dim)]
        self.step_num = 0
        # Initialize pursuers
        return self.step((0,0))[0]

    @property
    def is_terminal(self):
        if self._timesteps >= self.timestep_limit:
            return True
        return False


    def get_obs(self,agent):
        # return np.concatenate()
        return self.states

    def step(self, action):
        # print(action)
        obslist = []
        for agent in self.agents:
            obslist.append(self.get_obs(agent))
        if type(action) != int:
            action = action.argmax()

        right = self.states.index(max(self.states))
        # rewards = [self.states[action]]
        if action == right:
            rewards = [1]
        else:
            rewards = [-1]
        self.states[action] += 1
        self.step_num += 1
        if self.step_num > self.episode_length:
            done = True
            print(self.step_num)
        else:
            done = False
        info = None
        return obslist, rewards, done, info


class Env_mec:
    """
    当前思路的问题在于，agent在每个slot都只处理一个任务，并没有设置可以处理多任务的逻辑
    """

    def __init__(self, agent_num,agent_compu_cap, agent_BW_cap):
        self.agent_num = agent_num
        # 环境的observation是每个agent的observation的拼接，而每个agent的observation是（剩余BW，剩余CPU，请求的datasize，请求需要的CPU）
        self.state_dim = 4 * agent_num
        self.action_dim = 2 * agent_num
        self.agents = []
        self.reward_mech = 'global'
        for i in range(self.agent_num):
            self.agents.append(Agent_mec(i,agent_compu_cap,agent_BW_cap))
        self.episode_length = 20000
        # 下面这一句需要仔细调整
        self.states = [(np.random.rand(4) for i in range(self.agent_num))]
        print("initial states:%s",self.states)
        self.step_num = 0
        self._timesteps = 0
        self._is_terminal = False
        self._timesteps_limit = 1000
    def terminate(self):
        return self._is_terminal
    def current_timestep(self):
        return self._timesteps

    def timestep_update(self):
        self._timesteps += 1
        self.step_num += 1
        if self._timesteps >self._timesteps_limit:
            self._is_terminal = True
        for sys_agent in self.agents:
            sys_agent.timestep_update()
        return None

    def get_param_values(self):
        return self.__dict__

    def reset(self):
        # Initialize
        self.step_num = 0
        self._timesteps = 0
        self._is_terminal = False
        obser_list = []
        for agent in self.agents:
            obser_list.append(agent.reset())
        return obser_list

    # @property
    # def is_terminal(self):
    #     if self._timesteps >= self.timestep_limit:
    #         return True
    #     return False


    def get_obs(self,agent):
        # return np.concatenate()
        return self.states

    def step(self, action):
        """
        每经过一个time slot（一个OFDM slot的时长是0.5ms）,就对环境观测、并进行决策，根据决策结果更新系统状态，具体动作包括：
        处理queued_task队首的任务，为该任务挂上资源占用，判断其reward然后，决定要不要将其放进processing_task_list
        检查agent的processing_task_list中的任务是否释放资源，若有任务结束，释放资源的则reward+1 （可以将其写在timestep方法中）
        处理任务队列队首的任务，检查分配给该任务的资源，考虑是否将其加入到processing_task_list，
        让每个agent更新生成下一slot的任务，并将其添加到任务队列末尾
        """
        self.timestep_update()
        obslist = []
        rewards = []
        #取出各个agent的action(BW,CPU),以其为key值，作为字典存放在agents_action列表里
        agents_action = []
        for i,agent in enumerate(self.agents):
            actions = {}
            actions['BW'] = action[i*self.agent_num]
            actions['CPU'] = action[i*self.agent_num+1]
            agents_action.append(actions)
            #各agent的action已就绪
            print("observation %s of agent %s and current timestep is %d:"%(agent.get_obs(agents_action[i]),agent.__repr__(),self.current_timestep()))
            obslist.append(agent.get_obs(agents_action[i]))
            current_task = agent.queued_task[0]

            # 任务上传速率
            # 上传所需时间
            # 任务计算速率
            # 计算所需时间
            reward = current_task.reward()
            rewards.append(reward)
            #agent.processing_task_list.append(current_task)
            agent.queued_task.remove(current_task)
            #处理agent.processing_task_llist中的任务，释放资源
            for task in agent.processing_task_list:
                # 每个任务所需时间减一
                agent.resource_release(task)
            agent.processing_task_list.append(current_task)
        done = self._is_terminal
        info = None
        return obslist, rewards, done, info

class Action_policy:
    def random_policy(self,BW,CPU):
        actions = []
        actions.append(BW*random.random())
        actions.append(CPU*random.random())
        return actions


if __name__ == "__main__":
    env = Env_mec(1,1,1)
    act = Action_policy()
    env.reset()
    steps = 0
    while True:
        available_BW = []
        available_CPU = []
        actions = []
        for agent in env.agents:
            available_BW.append(agent.bw_left)
            available_CPU.append(agent.com_left)
            actions.extend(act.random_policy(available_BW[-1],available_CPU[-1]))
        obs,reward,done,_ = env.step(actions)
        steps+=1
        if done:
            break
    print("take %d steps"%steps)

