from abc import abstractmethod
from copy import copy, deepcopy
import math
from random import randrange
from abstract_agent import Agent
import backgammon_helper as bg
import numpy as np

possible_dies = np.array([(1,2),(1,3),(1,4),(1,5),(1,6),(2,3),(2,4),(2,5),(2,6),(3,4),(3,5),(3,6),(4,5),(4,6),(5,6), (1,1),(2,2),(3,3),(4,4),(5,5),(6,6)]) 

class Node:
    def __init__(self, gamestate, player, parent=None,):
        self.gamestate = gamestate
        self.player = player
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_value = 0.0

    def get_value(self):
        return self.total_value / (self.visits + 1e-6) # Ensure Non-Zero

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        if bg.game_over(self.gamestate):
            return True
        return False

    def backpropagate(self, reward):
        """Backpropagate reward"""
        self.visits += 1
        self.total_value += reward
        if self.parent: # backpropagate until the Root
            self.parent.backpropagate(reward)
            

    @abstractmethod
    def get_legal_actions(self,gamestate, dice):
        ...
    
    @abstractmethod
    def expand(self):
        ...
    
    @abstractmethod
    def best_child(self, c_param=1.4):
        ...

class DecisionNode(Node):
    def __init__(self, gamestate, dice, player, parent=None,):
        super().__init__(gamestate,player,parent)
        self.dice = dice
        self.untried_actions = self.get_legal_actions()

    def get_legal_actions(self):
        move_sequences, _ = bg.get_unique_legal_move_sequences(self.gamestate, self.dice, self.player)

        return list(move_sequences)
    
    def expand(self):
        """Expand untried action."""
        move_sequence = self.untried_actions.pop()

        next_gamestate = bg.get_next_board_total(self.gamestate,move_sequence,self.player)

        child_node = ChanceNode(next_gamestate, move_sequence, self.player, parent=self)
        self.children.append(child_node)     

        return child_node

    def best_child(self, c_param=1.4):
        """Select child with best UCB1 value."""
        choices = [
            (c.get_value() + # Exploit
            c_param * math.sqrt(math.log(self.visits + 1) / (c.visits + 1e-6)), # Explore (Ensure Non-Zero)
            c)
            for c in self.children
        ]
        if choices:
            _, c = max(choices,key= lambda x:x[0])
        return c
    
    def evaluate(self, n, max_depth):
        return None


class ChanceNode(Node):
    def __init__(self, gamestate, move, player, parent=None,):
        super().__init__(gamestate, player, parent)
        self.move = move
        self.untried_actions = self.get_legal_actions()
    
    def get_legal_actions(self):
        global possible_dies
        return list(possible_dies)
    
    def expand(self):
        dice = self.untried_actions.pop()
        child_node = DecisionNode(self.gamestate,dice,self.player,self)
        self.children.append(child_node)

        return child_node

    def best_child(self, c_param=1.4):
        L = len(self.children)
        deficits = [(self.visits*2 - c.visits, c) if c.dice[0]==c.dice[1] else (self.visits*1 - c.visits, c) for idx, c in enumerate(self.children)] # visits * 1/36 or 2/36, no point in / by 36
        _, c = max(deficits, key= lambda x: x[0])
        return c
    
    def evaluate(self, n, max_depth):
        def get_random_dice():
            return (randrange(1,7),randrange(1,7))
               
        total_value = 0
        for i in range(n):
            temp_gamestate = copy(self.gamestate)
            player = self.player
            d = 0

            while not (bg.game_over(temp_gamestate)) or d < max_depth:
                valid_action_sequences,_ = bg.get_legal_move_sequences(temp_gamestate,get_random_dice(),player)
                if valid_action_sequences:
                    size = len(valid_action_sequences)
                    action = randrange(0,size)  
                    temp_gamestate = bg.get_next_board_total(temp_gamestate, valid_action_sequences[action], player)
                player = -player
                d+=1

            total_value += bg.get_winner(temp_gamestate) # -1, 0 or 1
        return (total_value/n)
    
class Agent_MCTS(Agent):

    def __init__(self,num_simulations = 50,num_rollouts=20 ,c_puct=1.4,temp=1,max_depth=50):
        self.num_simulations = num_simulations
        self.num_rollouts = num_rollouts
        self.c_puct = c_puct
        self.temp = temp
        self.max_depth = max_depth

    def select_action_with_temperature(self, root, temperature=0):
        visits = np.array([child.visits for child in root.children])
        
        if temperature <= 1e-8:  # greedy
            best = np.argmax(visits)
            probs = np.zeros_like(visits)
            probs[best] = 1.0
            return root.children[best].action, probs
        # temperature scaling
        visits_scaled = visits ** (1.0 / temperature)
        probs = visits_scaled / (np.sum(visits_scaled))
        # sample according to the softmax probabilities

        idx = np.random.choice(len(root.children), p=probs)
        return root.children[idx].move, probs

    def get_next_move(self,gamestate, dice: tuple[int, int], player:int):
        
        root = DecisionNode(gamestate,dice, player)
        i = 0
        while i < self.num_simulations:
            

            node = root

            # Select
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.c_puct)
            
            # Expand
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

                
            # Simulate (rollout)
            reward = node.evaluate(self.num_rollouts, self.max_depth)
            # Backpropagate
            if reward:
                print(f"R{i}: ",end="")
                i+=1
                node.backpropagate(reward)
        
        print([child.visits for child in root.children])
        action, probs = self.select_action_with_temperature(root,self.temp)

        return action