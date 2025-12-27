from __future__ import annotations

from collections import defaultdict
from copy import copy, deepcopy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from networks.AlphaZeroNetwork import AlphaZeroNet
from src.MoveSequence import MoveSequence
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from BackgammonUtils import BackgammonUtils

# node_type = 0: Action node, no double
# node_type = 1: Action node, double, first move
# node_type = 2: Action node, double, second move
# node_type = 3: Chance node


@dataclass
class MCTSNode:
    board: np.ndarray
    player: int
    node_type: int
    parent: Optional[MCTSNode] = None
    node_action: Optional[list] = None
    children: Dict[List[list], MCTSNode] = field(default_factory=dict)
    legal_moves: Optional[List[list]] = None

    v: float = 0.0
    P: defaultdict = field(default_factory=lambda: defaultdict(float))
    N: defaultdict = field(default_factory=lambda: defaultdict(int))
    W: defaultdict = field(default_factory=lambda: defaultdict(float))
    Q: defaultdict = field(default_factory=lambda: defaultdict(float))

    def ucb_best(self, exploration: float = 1.4) -> float:
        if self.parent:
                self_visits = self.parent.N[BackgammonUtils._movesequence_key(self.node_action)]
        else:
            self_visits = sum(self.N.values())

        best_move = max(self.legal_moves, key=lambda m: 
            self.Q[BackgammonUtils._movesequence_key(m)] + # Child Value
            exploration * self.P[BackgammonUtils._movesequence_key(m)] * math.sqrt(self_visits)/(1+self.N[BackgammonUtils._movesequence_key(m)]) # Child Prior * sqrt(My Visits) / Child Visits
        )
        return best_move
    
    def chance_best(self) -> float:
        best_move = min(self.legal_moves, key=lambda m: 2*self.N[BackgammonUtils._movesequence_key(m)] if BackgammonUtils._movesequence_key(m) in [[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]] else self.N[BackgammonUtils._movesequence_key(m)])
        return best_move
    
    def best_child(self):
        if self.node_type == 3:
            best_move = self.chance_best()
        else:
            best_move = self.ucb_best()

        best_child = self.children.get(BackgammonUtils._movesequence_key(best_move),None)
        return best_child, best_move

    def best_move(self):
        if self.node_type == 1:
            first_best_move = max(
                self.legal_moves,
                key=lambda m: (self.N[tuple(m)], self.Q[tuple(m)])
            )
            best_child = self.children.get(tuple(first_best_move),None)
            if best_child and best_child.children:
                second_best_move = best_child.best_move()
            elif BackgammonUtils.game_over(best_child.board):
                second_best_move = []
            else:
                raise Exception("Too Few Simulations")
            


            return list(first_best_move) + list(second_best_move)
        else:
            best_move = max(
                self.legal_moves,
                key=lambda m: (self.N[tuple(m)], self.Q[tuple(m)])
            )
            return list(best_move)


    def __str__(self):
        return f"MCTSNode(Type: {self.node_type}, children: {len(self.children.values())}, Visits: {sum(self.N.values())}, Value: {sum(self.W.values())}, Action {self.node_action})\n"

    def __repr__(self) -> str:
        return str(self)
    
    def clear(self):
        stack = [self]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())

            node.parent = None
            node.children.clear()

class AlphaZeroAgent(AgentBase):
    nodes = 0

    def __init__(
        self,
        colour: Colour,
        simulations: int = 300,
        dirichlet_alpha = 0.03,
        dirichlet_epsilon = 0.25,
        model = None,
        random_seed = None,
    ):
        super().__init__(colour)
        self.simulations = simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        torch.manual_seed(0)
        if model:
            self.model = model # Use Loaded Model
        else:
            self.model = AlphaZeroNet()
            try:
                self.model.load_state_dict(torch.load("models/best_model.pth",weights_only=True))
            except Exception:
                # print("Model not Found")
                # print("Model Initialized with Random Weights")
                pass

    def make_move(
        self,
        board: Board,
        dice: list[int, int],
        opp_move: MoveSequence | None,
    ) -> MoveSequence:
        root_board = BackgammonUtils.get_internal_board(board)
        player = self._player_id()

        if dice[0] != dice[1]:
            self.root = MCTSNode(board=root_board, player=player, node_type=0, node_action=dice)
        else:
            self.root = MCTSNode(board=root_board, player=player, node_type=1, node_action=dice)
        self.root.legal_moves = BackgammonUtils.TwoMove_get_legal_move_sequences(root_board, list(dice), player)

        self._add_dirichlet_noise_to_root()

        for i in range(self.simulations):
            node = self._select(self.root)
            value = self._evaluate(node)
            self._backpropagate(node, value)

        # stack = [self.root]
        # while stack:
        #     node = stack.pop()
        #     print(node)
        #     for key, val in node.children.items():
        #         stack.append(val)
        # return None
    
        if self.root.children:
            best_move = self.root.best_move()
        else: 
            raise Exception("Too Few Simulations Run")

        self.root.clear()
        self.root = None
        return BackgammonUtils.get_external_movesequence(best_move)

    # ---------------- MCTS phases ---------------- #

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not BackgammonUtils.game_over(node.board):
            best_child, best_move = node.best_child()
            if not best_child:
                return self._expand(node, best_move)
            node = best_child
        return node

    def _expand(self, node: MCTSNode, best_move) -> MCTSNode:
        # node_type = 0: Action node, no double
        # node_type = 1: Action node, double, First
        # node_type = 2: Action node, double, Second
        # node_type = 3: Chance node
        self.nodes+=1
        action_made = best_move

        if node.node_type == 0:     
            # Action node, not Double -> next 3
            # self  node_action => Dice
            # Child node_action => Move Sequence
            #       action_made => Move Sequence

            next_board = node.board.copy()

            BackgammonUtils.do_next_board_total(
            next_board , action_made, node.player
            )

            child = MCTSNode(
                board           = next_board,
                player          = node.player,
                parent          = node,
                node_action     = action_made,
                node_type       = 3
            )

            child.legal_moves = [[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[2,2],[2,3],[2,4],[2,5],[2,6],[3,3],[3,4],[3,5],[3,6],[4,4],[4,5],[4,6],[5,5],[5,6],[6,6]]

        elif node.node_type == 1:   
            # Action node, double, first move -> next 2
            # self  node_action => Dice
            # Child node_action => Move Sequence
            #       action_made => Move Sequence

            next_board = node.board.copy()

            BackgammonUtils.do_next_board_total(
            next_board , action_made, node.player
            )

            child = MCTSNode(
                board           = next_board,
                player          = node.player,
                parent          = node,
                node_action     = action_made,
                node_type       = 2
            )

            child.legal_moves = BackgammonUtils.TwoMove_get_legal_move_sequences(
                    next_board, node.node_action, child.player
                )

        elif node.node_type == 2:   
            # Action node, double, second move -> next 3
            # self  node_action => Move Sequence
            # Child node_action => Move Sequence
            #       action_made => Move Sequence
            next_board = node.board.copy()

            BackgammonUtils.do_next_board_total(
            next_board , action_made, node.player
            )

            child = MCTSNode(
                board           = next_board,
                player          = node.player,
                parent          = node,
                node_action     = action_made,
                node_type       = 3
            )

            child.legal_moves = [[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[2,2],[2,3],[2,4],[2,5],[2,6],[3,3],[3,4],[3,5],[3,6],[4,4],[4,5],[4,6],[5,5],[5,6],[6,6]]

        elif node.node_type == 3:   
            # Chance node  -> next 0 or 1
            # self  node_action => Move Sequence
            # Child node_action => Dice
            #       action_made => Dice
            
            if action_made[0] == action_made[1]: # chance node, therefor node_action is dice
                nt = 1
            else:
                nt = 0
            
            child = MCTSNode(
                board           = node.board, # no move made, so children can share board object
                player          =-node.player, # swap player on exiting chance node
                parent          = node,
                node_action     = action_made,
                node_type       = nt
            )

            child.legal_moves = BackgammonUtils.TwoMove_get_legal_move_sequences(
                    node.board, action_made, child.player
                )

        node.children[tuple(action_made)] = child
        return child

    def _evaluate(self, node: MCTSNode) -> float:
        if BackgammonUtils.game_over(node.board):
            winner = BackgammonUtils.has_won(node.board)
            return 1.0 if winner == node.player else -1.0
        return self._evaluate_node(node)
        # return BackgammonUtils.heuristic_evaluation(node.board,node.player)
        
    def _backpropagate(self, node: MCTSNode, value: float):
        while node.parent:

            if not (node.node_action in node.parent.legal_moves):
                raise Exception
            
            key = tuple(node.node_action)
            node.parent.N[key] += 1
            node.parent.W[key] += value
            node.parent.Q[key] = node.parent.W[key]/node.parent.N[key]

            if node.node_type == 0: # Action node, not Double -> parent 3
                value = -value
                node = node.parent

            elif node.node_type == 1: # Action node, Double, First -> parent 3
                value = -value
                node = node.parent

            elif node.node_type == 2: # Action node, Double, Second -> parent 2
                value = value
                node = node.parent

            elif node.node_type == 3: # Chance node -> parent 0 or 2
                value = value
                node = node.parent

    # ---------------- Helpers ---------------- #

    def _player_id(self) -> int:
        return 1 if self.colour == Colour.RED else -1
    
    def _get_node_dice(self,node:MCTSNode):
        if node.node_type == 0:
            return node.node_action
        elif node.node_type == 1:
            return node.node_action
        elif node.node_type == 2:
            if node.parent:
                return self._get_node_dice(node.parent)
            else:
                return [0,0]
        elif node.node_type == 3:
            if node.parent:
                return self._get_node_dice(node.parent)
            else:
                return [0,0]

    def _get_previous_board_dice_player(self,node:MCTSNode):
        if node.node_type == 0:
            if node.parent:
                return self._get_previous_board_dice_player(node.parent)
            else:
                return np.zeros_like(node.board),[0,0],node.player
        elif node.node_type == 1:
            if node.parent:
                return self._get_previous_board_dice_player(node.parent)
            else:
                return np.zeros_like(node.board),[0,0],node.player
        elif node.node_type == 2:
            return node.board,self._get_node_dice(node),node.player
        elif node.node_type == 3:
            return node.board,self._get_node_dice(node),node.player

    def _evaluate_node(self, node:MCTSNode):
        # encode board
        if node.parent:
            past_board, past_dice, past_player = self._get_previous_board_dice_player(node.parent)
        else:
            past_board = np.zeros_like(node.board)
            past_dice = [0,0]
            past_player =  node.player

        present_board = node.board
        present_dice = self._get_node_dice(node)
        present_player =  node.player

        encoded = BackgammonUtils.encode_board(present_board,present_dice,present_player,past_board,past_dice,past_player)

        # get value and policy
        with torch.no_grad():
            value, policy = self.model(encoded)
            value = float(value.item())
            policy = policy[0].cpu().numpy()

        if node.node_type != 3:
            pol = {}
            for move_sequence in node.legal_moves:
                decoded_move_sequence = [BackgammonUtils.decode_move(move,node.player) for move in move_sequence]
                idx = BackgammonUtils.get_prior_idx(decoded_move_sequence,present_dice)
                # pol[tuple(move_sequence)] = float(policy[idx])
                pol[tuple(move_sequence)] = 1

            # normalize
            s = sum(pol.values()) + 1e-12
            for k in pol:
                pol[k] /= s

            node.P = pol
        return value
    
    def _add_dirichlet_noise_to_root(self):
        """
        Add Dirichlet noise to the root node's policy. This mixes the original policy
        with a Dirichlet sample over legal moves.
        """

        # list legal moves in deterministic order to align with node.P keys

        # extract original probs in same order as legal
        L = len(self.root.legal_moves)

        orig_probs = np.array([self.root.P.get(tuple(m), 0.0) for m in self.root.legal_moves], dtype=np.float64)
        
        # normalize just in case
        s = orig_probs.sum()
        if s <= 0:
            # fallback uniform
            orig_probs = np.ones_like(orig_probs) / L
        else:
            orig_probs = orig_probs / s

        # sample dirichlet noise
        if self.dirichlet_alpha > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * L)
            mixed = (1 - self.dirichlet_epsilon) * orig_probs + self.dirichlet_epsilon * noise
        else:
            # self.dirichlet_alpha = 0 -> no noise
            mixed = orig_probs

        # write back into node.P for the legal moves
        for m, p in zip(self.root.legal_moves, mixed):
            self.root.P[tuple(m)] = float(p)

        # ensure normalization
        s2 = sum(self.root.P.values()) + 1e-12
        for k in self.root.P:
            self.root.P[k] = self.root.P[k] / s2