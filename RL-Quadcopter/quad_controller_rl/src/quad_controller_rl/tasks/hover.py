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
        max_force = 25.0
        max_torque = 1.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Hover(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 15.0  # secs
        self.target_z = 10.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_ox = 0.0
        self.target_oy = 0.0
        self.target_oz = 0.0
        self.takeoff_duration = 5.0

          # 记录阶段性变量
        self.episode=0
        self.accumulated_rewards=0 # 每个阶段的累积奖励
        self.total_rewards=[]  # 存储各阶段累积奖励的数组，长度与episode的个数对应

    def reset(self):
        # Nothing to reset; just return initial condition
        return Pose(
                position=Point(0.0, 0.0, np.random.normal(0.5, 0.1)),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def save_records(self):
        fileObject = open('/home/robond/catkin_ws/src/RL-Quadcopter/quad_controller_rl/src/quad_controller_rl/tasks/rewards05.txt', 'w') 
        for r in self.total_rewards:
            fileObject.write(str(r)+" ,") 
        fileObject.close()

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        state = np.array([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        done = False
        
        # 检查位置稳定性
        reward=-(pose.position.x+pose.position.y+pose.orientation.x+pose.orientation.y+pose.orientation.w)*100
        
        # Base action
        action = np.zeros(6)

        # 在5秒之内，奖励函数的设定跟takeoff一致
        if timestamp<=self.takeoff_duration:
            reward+=-min(abs(self.target_z - pose.position.z),
                          20.0)  
            
            # before target height,apply no torque
            ac=self.agent.step(state, reward, done)
            action=np.array([ac[0],ac[1],ac[2],0,0,0])
        else:
            # 5秒之后如果还不能达到目标高度，起飞失败，结束阶段，并给以较大的惩罚
            if pose.position.z < self.target_z:
                reward+=-30
                done = True
            # 5秒之后，需要保持高度稳定，如果高度跟目标高度差小于0.5,奖励加30，否则减30
            if abs(self.target_z - pose.position.z) < 0.5:
                reward+=30
            else:
                reward+=-50
       
            action=self.agent.step(state, reward, done)
            # 检查阶段是否结束
            if timestamp>self.max_duration:
                done=True


        if done:
            self.total_rewards.append(self.accumulated_rewards)
            self.accumulated_rewards=0
            self.save_records()
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

