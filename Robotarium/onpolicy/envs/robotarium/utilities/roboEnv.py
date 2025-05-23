import copy
import rps.robotarium as robotarium
import numpy as np
import random
import time

#This file should stay as is when copied to robotarium_eval but local imports must be changed to work with training!
from onpolicy.envs.robotarium.utilities.controller import *
from onpolicy.envs.robotarium.utilities.misc import *

class roboEnv:
    def __init__(self, agents, args):
        self.args = args
        self.agents = agents
        if "barrier_certificate" in self.args.__dict__.keys():
            self.controller = Controller(self.args.barrier_certificate)
        else:
            self.controller = Controller()
        self.first_run = True 
        self.episodes = 0
        self.errors = {"collision":0, "boundary":0}

        # Figure showing and visualizing
        self.visualizer = agents.visualizer
        

    def reset(self, show_figure = False):
        '''
        Reset the environment
        '''
        self.episodes += 1
        self.visualizer.show_figure = show_figure
        # if self.args.show_figure_frequency == -1 or self.episodes % self.args.show_figure_frequency > 0:
        #     self.visualizer.show_figure = False
        # else:
        #     self.visualizer.show_figure = True
        self._create_robotarium()

    def step(self, actions_):
        '''
        Take a step into the environment given the action
        '''
        goals_ = self.agents._generate_step_goal_positions(actions_)
        dist_travelled = np.zeros((self.agents.num_robots))
        frames = []

        #This should be renamed but this returns the reason the episode stopped
        #'' means the episode ended successfully, 'boundary' means the episode ended due to boundary, 'collision' means there was a collision
        message = '' 


        # Considering one step to be equivalent to update_frequency iterations
        for iterations in range(self.args.update_frequency):
            # Get the actual position of the agents
            self.agents.agent_poses = self.robotarium.get_poses()
            if self.previous_pose is not None:
                dist_travelled += np.linalg.norm(self.agents.agent_poses[:2, :] - self.previous_pose[:2, :], axis=0)
            
            # Saving the pose from the previous iteration
            self.previous_pose = copy.deepcopy(self.agents.agent_poses)

            # Uses the robotarium commands to get the velocities of each robot   
            # Only does this once every 10 steps because otherwise training is really slow 
            if iterations % 15 == 0 or self.args.robotarium:                    
                velocities = self.controller.set_velocities(self.agents.agent_poses, goals_)
                self.robotarium.set_velocities(np.arange(self.agents.num_robots), velocities)
            
            # if self.visualizer.show_figure:
            self.visualizer.update_markers(self.robotarium, self.agents)

                # if self.args.save_gif and self.counter % self.args.gif_frequency == 0:
                #     fig = self.robotarium.figure
                #     fig.canvas.draw()
                #     frame = np.array(fig.canvas.renderer.buffer_rgba())
                #     frames.append(frame)
                
                # self.counter += 1

            self.robotarium.step()
            #If checking for violations (boundary and collision) then
            #   Check if there has been a violation ever and then check if the number of times that violation has occurred has increased since the last timestep
            if self.args.penalize_violations:
                if isinstance(self.robotarium._errors.get('collision'), int):
                    self.robotarium._errors['collision'] = {'default': self.robotarium._errors['collision']}
                if isinstance(self.errors.get('collision'), int):
                    self.errors['collision'] = {'default': self.errors['collision']}

                # 로직 수행
                if 'collision' in self.robotarium._errors and (
                        'collision' not in self.errors or
                        sum(self.robotarium._errors['collision'].values()) > sum(self.errors['collision'].values())
                ):
                    message = 'collision'

                if isinstance(self.robotarium._errors.get('boundary'), int):
                    self.robotarium._errors['boundary'] = {'default': self.robotarium._errors['boundary']}
                if isinstance(self.errors.get('boundary'), int):
                    self.errors['boundary'] = {'default': self.errors['boundary']}
                if 'boundary' in self.robotarium._errors and ('boundary' not in self.errors or sum(self.robotarium._errors['boundary'].values()) > sum(self.errors['boundary'].values())):
                    if message == '':
                        message = 'boundary'
                    else:
                        message += "_boundary"
                self.errors = copy.deepcopy(self.robotarium._errors)
                if message != '':
                    dist_travelled += np.linalg.norm(self.agents.agent_poses[:2, :] - self.previous_pose[:2, :], axis=0)
                    return message, dist_travelled, frames
        
        return "", dist_travelled, frames
    
    def _create_robotarium(self):
        '''
        Creates a new instance of the robotarium
        Uses the scenario's visualizer object to generate the plots
        '''
        # Initialize agents and tracking variables
        if self.first_run:
            self.first_run = False
        else:
            del self.robotarium

        self.robotarium = robotarium.Robotarium(number_of_robots= self.agents.num_robots, show_figure = self.visualizer.show_figure, \
                                                initial_conditions=self.agents.agent_poses, sim_in_real_time=self.args.real_time)
        self.agents.agent_poses = self.robotarium.get_poses()    
        self.robotarium.step()

        # if self.visualizer.show_figure:
        self.visualizer.initialize_markers(self.robotarium, self.agents)
        
        self.previous_pose = None
        self.counter = 0

    def __del__(self):
        self.robotarium.call_at_scripts_end()

