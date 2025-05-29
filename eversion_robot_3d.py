import gymnasium as gym
from gymnasium import spaces
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class EversionRobot3D(gym.Env):
    def __init__(self, obs_use):
        super(EversionRobot3D, self).__init__()
        self.MAX_EPISODE = 100
        self.x_threshold = 5
        self.use_obstacle = obs_use
        
        if self.use_obstacle:
            high = np.array([self.x_threshold, self.x_threshold, self.x_threshold, 
                            self.x_threshold, self.x_threshold, self.x_threshold, 2.0],
                        dtype=np.float32)
        else:
            high = np.array([self.x_threshold, self.x_threshold, self.x_threshold, 
                           self.x_threshold, self.x_threshold, self.x_threshold],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(7)  # Actions for 3D control
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.steps_left = self.MAX_EPISODE
        self.low = [-1.5, 0.5, 0]
        self.high = [1.5, 3.0, 1.5]
        self.x_target = [random.uniform(self.low[0], self.high[0]), 
                        random.uniform(self.low[1], self.high[1]),
                        random.uniform(self.low[2], self.high[2])]
        
        if self.use_obstacle:
            self.state = [0, 0, 0, self.x_target[0], self.x_target[1], self.x_target[2], 0]
        else:
            self.state = [0, 0, 0, self.x_target[0], self.x_target[1], self.x_target[2]]
            
        self.init_length = 0.1
        self.length = self.init_length
        
        # Changed from kappa/phi to separate curvatures for each axis
        self.kappa_x = 0  # Curvature for bending in x direction (left/right)
        self.kappa_z = 0  # Curvature for bending in z direction (up/down)
        
        self.delta_length = 0.1
        self.delta_kappa = 0.1
        
        self.segment_num = 1
        self.segment_num_max = 5
        self.length_max = 1.5
        self.length_array = [self.length]
        self.kappa_x_array = [self.kappa_x]
        self.kappa_z_array = [self.kappa_z]
        self.T_static = np.eye(4)
        self.safety_param = 2
        self.safety_penalty = 10

        # Obstacle's centre position and radius (adjusted for Y-axis growth)
        if self.use_obstacle:
            self.obs_center = []
            self.obs_center.append(np.array([0.0, 3.5, 0.0]))
            self.obs_center.append(np.array([-1.1, 1.5, 0.5]))
            self.obs_center.append(np.array([0.7, 1.0, 0]))
            self.radius = []
            for i in range(0, len(self.obs_center)):
                self.radius.append(0.2)

    def constant_curvature_3d(self, kappa_x, kappa_z, length):
        """
        Calculate tip position for 3D constant curvature segment
        kappa_x: curvature in x direction (bending left/right)
        kappa_z: curvature in z direction (bending up/down)
        """
        # Total curvature magnitude
        kappa_total = np.sqrt(kappa_x**2 + kappa_z**2)
        
        if kappa_total != 0:
            # Unit vector in bending direction
            bending_dir_x = kappa_x / kappa_total
            bending_dir_z = kappa_z / kappa_total
            
            # Arc length parameter
            s = kappa_total * length
            
            # Position calculation for 3D constant curvature (growing along Y-axis)
            x_tip = bending_dir_x * (1 - np.cos(s)) / kappa_total
            y_tip = np.sin(s) / kappa_total  # Main growth direction
            z_tip = bending_dir_z * (1 - np.cos(s)) / kappa_total
        else:
            # Straight segment growing along Y-axis
            x_tip = 0
            y_tip = length
            z_tip = 0
            
        return x_tip, y_tip, z_tip

    def get_transformation_matrix(self, kappa_x, kappa_z, length):
        """
        Get transformation matrix for 3D constant curvature segment
        Growing along Y-axis with bending in X and Z directions
        """
        # Total curvature magnitude
        kappa_total = np.sqrt(kappa_x**2 + kappa_z**2)
        
        if kappa_total != 0:
            # Unit vector in bending direction
            bending_dir_x = kappa_x / kappa_total
            bending_dir_z = kappa_z / kappa_total
            
            # Arc parameter
            s = kappa_total * length
            cos_s = np.cos(s)
            sin_s = np.sin(s)
            
            # Position of tip (growing along Y-axis)
            x_tip = bending_dir_x * (1 - cos_s) / kappa_total
            y_tip = sin_s / kappa_total  # Main growth direction
            z_tip = bending_dir_z * (1 - cos_s) / kappa_total
            
            # Orientation matrix for bending in XZ plane while growing along Y
            if abs(kappa_x) > abs(kappa_z):
                # Primary bending in x direction
                R = np.array([
                    [cos_s, -np.sign(kappa_x) * sin_s, 0],
                    [np.sign(kappa_x) * sin_s, cos_s, 0],
                    [0, 0, 1]
                ])
            else:
                # Primary bending in z direction
                R = np.array([
                    [1, 0, 0],
                    [0, cos_s, np.sign(kappa_z) * sin_s],
                    [0, -np.sign(kappa_z) * sin_s, cos_s]
                ])
                
            # Construct homogeneous transformation matrix
            A = np.eye(4)
            A[0:3, 0:3] = R
            A[0:3, 3] = [x_tip, y_tip, z_tip]
            
        else:
            # Straight segment growing along Y-axis
            A = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, length],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
        return A

    def static_segment(self, kappa_x_array, kappa_z_array, length_array, segment_index):
        T_multi = np.eye(4)
        for i in range(0, segment_index):
            kappa_x = kappa_x_array[i]
            kappa_z = kappa_z_array[i]
            length = length_array[i]
            T_single = self.get_transformation_matrix(kappa_x, kappa_z, length)
            T_multi = T_multi @ T_single
        return T_multi

    def pose_segment(self, segment_index):
        T_prior_segment = self.static_segment(self.kappa_x_array, self.kappa_z_array, self.length_array, segment_index)
        indeks_maks = math.floor(self.length_array[segment_index]/self.delta_length)
        
        if indeks_maks != 0:
            kappa_x_increment = self.kappa_x_array[segment_index]/indeks_maks
            kappa_z_increment = self.kappa_z_array[segment_index]/indeks_maks
        else:
            kappa_x_increment = 0
            kappa_z_increment = 0
            
        x_array = []
        y_array = []
        z_array = []
        
        for i in range(0, indeks_maks):
            length_increment = self.delta_length * i
            T_single = self.get_transformation_matrix(kappa_x_increment * i, kappa_z_increment * i, length_increment)
            transform_xyz = T_prior_segment @ T_single
            x_tip, y_tip, z_tip = transform_xyz[0, 3], transform_xyz[1, 3], transform_xyz[2, 3]
            x_array.append(x_tip)
            y_array.append(y_tip)
            z_array.append(z_tip)
            
        return x_array, y_array, z_array

    def check_collision(self):
        collision = False
        x_array, y_array, z_array = self.pose_segment(len(self.length_array)-1)
        
        for i in range(0, len(self.obs_center)):
            for j in range(0, len(x_array)):
                distance_vect = np.array([x_array[j] - self.obs_center[i][0],
                                         y_array[j] - self.obs_center[i][1],
                                         z_array[j] - self.obs_center[i][2]])
                distance_scalar = np.linalg.norm(distance_vect)
                if distance_scalar <= self.radius[i]:
                    collision = True
                    break
                
            if collision:
                break
        
        return collision

    def check_safety(self):
        danger = False
        x_array, y_array, z_array = self.pose_segment(len(self.length_array)-1)
        
        for i in range(0, len(self.obs_center)):
            for j in range(0, len(x_array)):
                distance_vect = np.array([x_array[j] - self.obs_center[i][0],
                                         y_array[j] - self.obs_center[i][1],
                                         z_array[j] - self.obs_center[i][2]])
                distance_scalar = np.linalg.norm(distance_vect)
                if distance_scalar <= self.safety_param * self.radius[i]:
                    danger = True
                    break
                
            if danger:
                break
        
        return danger

    def obstacle_vector(self, dist_to_goal):
        limit_obs = 0.5
        gain_obs = 3.0
        x_array, y_array, z_array = self.pose_segment(len(self.length_array)-1)
        min_distance = 100
        distance_vect_min = np.array([min_distance, min_distance, min_distance])
        
        for i in range(0, len(self.obs_center)):
            for j in range(0, len(x_array)):
                distance_vect = np.array([x_array[j] - self.obs_center[i][0],
                                         y_array[j] - self.obs_center[i][1],
                                         z_array[j] - self.obs_center[i][2]])
                distance_scalar = np.linalg.norm(distance_vect)
                if distance_scalar < min_distance:
                    min_distance = distance_scalar
                    distance_vect_min = distance_vect
        
        direction_obs = distance_vect_min / min_distance
        if np.linalg.norm(dist_to_goal) != 0:
            direction_goal = dist_to_goal / np.linalg.norm(dist_to_goal)
        else:
            direction_goal = dist_to_goal
            
        cos_angle = np.dot(direction_goal, direction_obs)
        if (min_distance - self.radius[i]) <= limit_obs and cos_angle > 0:
            distance_to_surface = min_distance - self.radius[i]
            avoidance_term = (1.0 / distance_to_surface - 1.0 / limit_obs) * gain_obs
        else:
            avoidance_term = 0
        
        return avoidance_term

    def step(self, action):
        self.act = action
        save_state = self.state
        
        # New Action definitions for 3D bending (growing along Y-axis):
        # 0: Increase length
        # 1: Decrease length
        # 2: Bend right (increase kappa_x)
        # 3: Bend left (decrease kappa_x)
        # 4: Bend up (increase kappa_z)
        # 5: Bend down (decrease kappa_z)
        # 6: No action
        
        if action == 0:
            self.length = self.length + self.delta_length
        elif action == 1:
            self.length = self.length - self.delta_length
        elif action == 2:  # Bend right
            self.kappa_x = self.kappa_x + self.delta_kappa
        elif action == 3:  # Bend left
            self.kappa_x = self.kappa_x - self.delta_kappa
        elif action == 4:  # Bend up
            self.kappa_z = self.kappa_z + self.delta_kappa
        elif action == 5:  # Bend down
            self.kappa_z = self.kappa_z - self.delta_kappa
        elif action == 6:  # No action
            pass
        
        # Update arrays when length exceeds current segment capacity
        if self.length > self.length_max and len(self.length_array) < self.segment_num_max:
            self.length_array[-1] = self.length_max
            self.length = 0
            self.kappa_x = 0
            self.kappa_z = 0
            self.length_array.append(self.length)
            self.kappa_x_array.append(self.kappa_x)
            self.kappa_z_array.append(self.kappa_z)
        elif self.length < 0 and len(self.length_array) > 1:
            self.length_array.pop()
            self.kappa_x_array.pop()
            self.kappa_z_array.pop()
            self.length = self.length_array[-1]
            self.kappa_x = self.kappa_x_array[-1]
            self.kappa_z = self.kappa_z_array[-1]
        else:
            self.length_array[-1] = self.length
            self.kappa_x_array[-1] = self.kappa_x
            self.kappa_z_array[-1] = self.kappa_z
        
        self.T_static = self.static_segment(self.kappa_x_array, self.kappa_z_array, self.length_array, len(self.kappa_x_array))
        x_tip, y_tip, z_tip = self.T_static[0, 3], self.T_static[1, 3], self.T_static[2, 3]
        
        if self.use_obstacle:
            safety_flag = self.check_safety()
            safety_obs = 1 if safety_flag else 0
            self.state = [x_tip, y_tip, z_tip, 
                         self.x_target[0], self.x_target[1], self.x_target[2], 
                         safety_obs]
        else:
            self.state = [x_tip, y_tip, z_tip, 
                         self.x_target[0], self.x_target[1], self.x_target[2]]
            
        boundary = (x_tip < -self.x_threshold or x_tip > self.x_threshold or 
                   y_tip < -self.x_threshold or y_tip > self.x_threshold or
                   z_tip < 0 or z_tip > self.x_threshold)
        
        error = np.array(self.state[0:3]) - np.array(self.x_target)
        
        if self.use_obstacle:
            self.flag_collision = self.check_collision()
            reward_safety = -1 * self.obstacle_vector(error)
            
        done = bool(
            boundary
            or self.steps_left < 0
            or self.length < 0
        )
        
        if self.use_obstacle:
            done = done or self.flag_collision
        
        if not done:
            reward = -np.linalg.norm(error)**2
            if self.use_obstacle:
                reward += reward_safety
        else:
            if self.length < 0:
                reward = -100000
            elif self.use_obstacle:
                if self.flag_collision:
                    reward = -100000
                else:
                    reward = 0
            else:
                reward = 0
                
        if not done:
            self.steps_left -= 1
            
        self.cur_reward = reward
        self.cur_done = done
        
        return np.array(self.state), reward, done, {},{}

    def reset(self,seed=None):
        self.x_target = [random.uniform(self.low[0], self.high[0]), 
                        random.uniform(self.low[1], self.high[1]),
                        random.uniform(self.low[2], self.high[2])]
        
        if self.use_obstacle:
            self.state = [0, 0, 0, 
                         self.x_target[0], self.x_target[1], self.x_target[2], 
                         0]
        else:
            self.state = [0, 0, 0, 
                         self.x_target[0], self.x_target[1], self.x_target[2]]
            
        self.steps_left = self.MAX_EPISODE
        self.length = self.init_length
        self.kappa_x = 0
        self.kappa_z = 0
        self.length_array = [self.length]
        self.kappa_x_array = [self.kappa_x]
        self.kappa_z_array = [self.kappa_z]
        self.T_static = np.eye(4)
        
        return np.array(self.state),{}

    def draw_segment(self, segment_index, ax):
        if segment_index % 2 == 0:
            color_plot = 'red'
        else:
            color_plot = 'blue'
            
        x_array, y_array, z_array = self.pose_segment(segment_index)
        
        # Only plot if we have points
        if len(x_array) > 0:
            ax.plot(x_array, y_array, z_array, color=color_plot, linewidth=2)
            ax.scatter(x_array, y_array, z_array, color=color_plot, s=20)

    def draw_obs(self, ax):
        for i in range(0, len(self.obs_center)):
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = self.obs_center[i][0] + self.radius[i] * np.outer(np.cos(u), np.sin(v))
            y = self.obs_center[i][1] + self.radius[i] * np.outer(np.sin(u), np.sin(v))
            z = self.obs_center[i][2] + self.radius[i] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='k', alpha=0.5)

    def render(self, mode='human'):
        # Create figure if it doesn't exist
        if not hasattr(self, 'fig'):
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            plt.ion()  # Turn on interactive mode
            plt.show()
        
        # Clear previous plot
        self.ax.clear()
        
        # Set axis limits and labels (adjusted for Y-axis growth)
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([0, 4])
        self.ax.set_zlim([0, 3])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Draw all segments
        for i in range(len(self.length_array)):
            self.draw_segment(i, self.ax)
        
        # Draw target
        self.ax.scatter(self.x_target[0], self.x_target[1], self.x_target[2], 
                    color='green', s=100, marker='*')
        
        # Draw obstacles if enabled
        if self.use_obstacle:
            self.draw_obs(self.ax)
        
        # Update title
        self.ax.set_title(f'3D Eversion Robot - Step {self.MAX_EPISODE - self.steps_left}')
        
        # Draw and pause briefly
        plt.draw()
        plt.pause(0.01)