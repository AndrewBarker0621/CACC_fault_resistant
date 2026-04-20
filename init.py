import numpy as np

class KinematicBicycleModel:
    def __init__(self, wheelbase=2.5, dt=0.05):
        self.L = wheelbase 
        self.dt = dt

    def step(self, state, control):
        """
        :param state: [x, y, theta, v]
        :param control: [a, delta]
        :return: next_state: [x_next, y_next, theta_next, v_next]
        """
        # x, y, theta, v = state
        # a, delta = control

        # # steering
        # delta = np.clip(delta, -np.radians(30), np.radians(30))  # ±30°

        # # state update
        # x_next = x + v * np.cos(theta) * self.dt
        # y_next = y + v * np.sin(theta) * self.dt
        # theta_next = theta + (v / self.L) * np.tan(delta) * self.dt
        # v_next = v + a * self.dt

        # simple_update without steering
        # """
        # :param state: [x, y, theta, v]
        # :param control: [a, delta]
        # :return: next_state: [x_next, y_next, theta_next, v_next]
        # """
        x, y, theta, v = state
        a, _ = control

        # # state update
        x_next = x + v * self.dt
        y_next = y
        theta_next = theta
        v_next = v + a * self.dt

        return [x_next, y_next, theta_next, v_next]
            
class PlatoonInitializer:
    def __init__(self, num_vehicles=5, spacing=10.0, initial_speed=10.0):
        """
        :param num_vehicles
        :param spacing
        :param initial_speed
        """
        self.num_vehicles = num_vehicles
        self.spacing = spacing
        self.initial_speed = initial_speed

    def initialize(self):
        """
        (x, y, theta, v)
        :return: states: list of [x, y, theta, v]
        """
        states = []
        lane_center_y = 1.5  # lane center

        for i in range(self.num_vehicles):
            x = 100.0 - i * (5.0 + self.spacing) 
            y = lane_center_y
            theta = 0.0
            v = self.initial_speed
            states.append([x, y, theta, v])
        print(states)
        return states
    
    def initialize_custom(self,
                          leader_x=100.0,
                          leader_v=10.0,
                          follower_gaps=None,
                          follower_speeds=None):
        """
        (d_i(0), v_i(0)) init
          - follower_gaps
          - follower_speeds
        return: states: list of [x, y, theta, v]
        """
        states = []
        lane_center_y = 1.5
        vehicle_length = 5

        N = self.num_vehicles
        assert N >= 1, "num_vehicles must be >= 1"

        if follower_gaps is None:
            follower_gaps = [10.0] * (N - 1)
        if follower_speeds is None:
            follower_speeds = [leader_v] * (N - 1)
        assert len(follower_gaps) == N - 1
        assert len(follower_speeds) == N - 1

        states = []
        # leader
        states.append([leader_x, lane_center_y, 0.0, float(leader_v)])

        # followers: cumulative placement
        x_prev = leader_x
        for i in range(1, N):
            d_i = float(follower_gaps[i - 1])
            v_i = float(follower_speeds[i - 1])
            # curr_pos = proceeding - (length + distance)
            x_i = x_prev - (vehicle_length + d_i)
            states.append([x_i, lane_center_y, 0.0, v_i])
            x_prev = x_i
        print(states)
        return states