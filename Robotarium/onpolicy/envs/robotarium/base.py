class BaseEnv(object):
    '''
    Template for what a Robotarium Environment must have
    Additionally, it must have the following class variables:
        num_agents, agent_poses, visualizer
    '''

    def get_action_space(self):
        # Must return the action space
        raise NotImplementedError()

    def get_observation_space(self):
        # Must return the observation space
        raise NotImplementedError()

    def step(self, actions_):
        # Must return [observations, rewards, done, info]
        raise NotImplementedError()

    def reset(self):
        # Must return an observation array
        raise NotImplementedError()

    def render(self, mode='human'):
        # This isn't really used in our environments
        pass

    def _generate_step_goal_positions(self, actions):
        # Must return goal locations for each agent
        raise NotImplementedError()


class BaseVisualization():
    '''
    Template for what the scenario's Visualize class must contain
    Additionally, it must also have show_figure as a class variable
    '''

    # How the robotarium's background gets reset at the beginning of each episode
    def initialize_markers(self, robotarium, agents):
        raise NotImplementedError()

    # How the robotarium's background changes at each robotarium step
    def update_markers(self, robotarium, agents):
        raise NotImplementedError()