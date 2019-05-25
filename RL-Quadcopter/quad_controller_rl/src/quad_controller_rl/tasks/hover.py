"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask


class Hover(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
        #print("Hover(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 23.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Hover(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.velocity = 0.0
        self.max_duration = 5.0  # secs
        self.target_z = 10.0
        self.takeoff_limit = 5.0
        self.height_limit=20.0
        self.target_hit=0
        self.last_time=0.0
        self.last_height=0.0

          # 记录阶段性变量
        self.episode=0
        self.accumulated_rewards=0 # 每个阶段的累积奖励
        self.total_rewards=[]  # 存储各阶段累积奖励的数组，长度与episode的个数对应
        self.actions=[]
        self.states=[]

    def reset(self):
        # Nothing to reset; just return initial condition
        return Pose(
                position=Point(0.0, 0.0, np.random.normal(0.5, 0.1)),
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def save_records(self):
        fileObject = open('/home/robond/catkin_ws/src/RL-Quadcopter/quad_controller_rl/src/quad_controller_rl/records/rewards_01.txt', 'w') 
        for r in self.total_rewards:
            fileObject.write(str(r)+" ,") 
        fileObject.close()
        
        
    def save_actions(self):
        fileObject = open('/home/robond/catkin_ws/src/RL-Quadcopter/quad_controller_rl/src/quad_controller_rl/records/actions_01.txt', 'w') 
        for action in self.actions:
            fileObject.write(str(action)+" ,")
        fileObject.close()
        
    def save_states(self):
        f=open('/home/robond/catkin_ws/src/RL-Quadcopter/quad_controller_rl/src/quad_controller_rl/records/states_01.txt', 'w')
        for state in self.states:
            f.write(str(state)+" ,")
        f.close()

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        done = False

        """更新速度，"""
        h = pose.position.z-self.last_height
        time = timestamp-self.last_time
        if time!=0:
            self.velocity=h/time
        self.last_height = pose.position.z
        self.last_time = timestamp
        
        """高度偏离惩罚"""
        reward=-abs(self.target_z-pose.position.z)*20
        
        """加速度惩罚"""
        if timestamp>1:
            reward-=abs(linear_acceleration.z)*100
            
        """高度目标奖励,速度惩罚"""
        if abs(pose.position.z-self.target_z) < 0.3:
            self.target_hit+=1
            reward+=10*(self.target_hit)**2
            reward-=abs(self.velocity)*100
            
        """阶段结束条件"""
        if pose.position.z > self.height_limit:
            done=True
        if timestamp > self.max_duration:
            done=True

        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        state = np.array([pose.position.x, pose.position.y, pose.position.z,
                          self.velocity,linear_acceleration.z,pose.orientation.x, pose.orientation.y, 
                          pose.orientation.z, pose.orientation.w])
        action=self.agent.step(state, reward, done)
        self.actions.append(action)
        self.states.append(self.velocity)
        
        if done:
            self.target_hit=0
            self.last_time=0.0
            self.last_height=0.0
            self.velocity=0.0
            self.total_rewards.append(self.accumulated_rewards)
            self.accumulated_rewards=0
            self.save_records()
            self.save_actions()
            self.save_states()
        else:
            self.accumulated_rewards+=reward

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done

