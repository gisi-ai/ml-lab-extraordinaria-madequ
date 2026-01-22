"""
Heuristic Alpha-Beta Chess Bot

A chess bot that uses alpha-beta pruning with heuristic evaluation functions
for positions that are not terminal states.
"""

from typing import Dict, Any, Callable

import chess

from adversarial_search.chess_game_state import ChessGameState
from adversarial_search.chess_problem import ChessProblem
from adversarial_search.game_algorithms import heuristic_alphabeta_search
from bots.chess_bot import ChessBot
from bots.features import get_position_features


def linear_weighted_heuristic(state: ChessGameState, player: chess.Color, weights: Dict[str, float]) -> float:
    """
    Generic linear weighted heuristic evaluation function.

    Collects position features, applies weighted linear combination, and clips the result.

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :param weights: Dictionary of feature weights (must sum to 1.0)
    :type weights: Dict[str, float]
    :return: Evaluation score (0 to 1)
    :rtype: float
    """
    board = state.get_board()
    features = get_position_features(board)
    score = weighted_linear_features(features, weights, player)
    return max(1e-2, min(1.0 - 1e-2, score))


def weighted_linear_features(features: Dict[str, float], weights: Dict[str, float], player: chess.Color) -> float:
    """
    Generic weighted linear combination of normalized features.

    :param features: Dictionary of features from get_position_features
    :type features: Dict[str, float]
    :param weights: Dictionary of feature weights (must sum to 1.0)
    :type weights: Dict[str, float]
    :param player: Player to evaluate for (chess.WHITE or chess.BLACK)
    :type player: chess.Color
    :return: Weighted evaluation score (0 to 1)
    :rtype: float
    :raises ValueError: If weights don't sum to 1.0 or contain invalid keys
    """
    # Validate weights
    normalized_features = {
        "material_diff_normalized",
        "positional_diff_normalized",
        "mobility_diff_normalized",
        "center_diff_normalized",
        "king_safety_diff_normalized",
        "development_diff_normalized"
    }

    # Check that all weight keys are valid normalized features
    invalid_keys = set(weights.keys()) - normalized_features
    if invalid_keys:
        raise ValueError(f"Invalid weight keys: {invalid_keys}. Valid keys: {normalized_features}")

    # Check that weights sum to approximately 1.0
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    # Check that all weights are non-negative
    negative_weights = {k: v for k, v in weights.items() if v < 0}
    if negative_weights:
        raise ValueError(f"Weights must be non-negative, got negative weights: {negative_weights}")

    # Calculate weighted score
    if player == chess.WHITE:
        score = sum(weights.get(feature, 0.0) * features[feature] for feature in normalized_features)
    else:
        # For black, invert the features (1.0 - feature_value)
        score = sum(weights.get(feature, 0.0) * (1.0 - features[feature]) for feature in normalized_features)

    return score


def h1(state: ChessGameState, player: chess.Color) -> float:
    """
    Heuristic 1: Equal-weighted evaluation.

    Uses uniform weights (1/6 ≈ 16.67% each) for all six normalized features.

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :return: Evaluation score (0 to 1)
    :rtype: float
    """
    equal_weight = 1.0 / 6
    weights = {
        "material_diff_normalized": equal_weight,
        "positional_diff_normalized": equal_weight,
        "mobility_diff_normalized": equal_weight,
        "center_diff_normalized": equal_weight,
        "king_safety_diff_normalized": equal_weight,
        "development_diff_normalized": equal_weight
    }
    return linear_weighted_heuristic(state, player, weights)


def h2(state: ChessGameState, player: chess.Color) -> float:
    """
    Heuristic 2

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :return: Evaluation score (0 to 1)
    :rtype: float
    """
    # TODO:
    board = state.get_board()
    resultados = get_position_features(board)
    weights = {
        "material_diff_normalized": 0.1,
        "positional_diff_normalized": 0.1,
        "king_safety_diff_normalized": 0.5,
        "development_diff_normalized": 0.3
    }
    return linear_weighted_heuristic(state, player, weights)



def h3(state: ChessGameState, player: chess.Color) -> float:
    """
    Heuristic 3:

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :return: Evaluation score (0 to 1)
    :rtype: float
    """
    # TODO:
    board = state.get_board()
    resultados = get_position_features(board)
    weights = {
        "material_diff_normalized": 0.5,
        "positional_diff_normalized": 0.2,
        "mobility_diff_normalized": 0.2,
        "king_safety_diff_normalized": 0.5
    }
    return linear_weighted_heuristic(state, player, weights)


class HeuristicAlphaBetaBot(ChessBot):
    """
    Chess bot using heuristic alpha-beta search with configurable parameters.
    """

    def __init__(self, eval_function: Callable[[ChessGameState, chess.Color], float]=h1,
                 max_depth: int = 3):
        """
        Initialize the heuristic alpha-beta bot.

        :param eval_function: Evaluation function (state, player) -> float
        :type eval_function: Callable[[ChessGameState, chess.Color], float]
        :param max_depth: Maximum search depth
        :type max_depth: int
        """
        # TODO: you may set your name as author
        super().__init__("HeuristicAlphaBetaBot", "Javier De Quadros")
        self.eval_function = eval_function
        self.max_depth = max_depth

    def get_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """
        Get the best move using heuristic alpha-beta search.

        :param board: Current chess position
        :type board: chess.Board
        :param time_limit: Maximum time to think in seconds
        :type time_limit: float
        :return: Best move found
        :rtype: chess.Move
        """
        # TODO:
        def cutoff_test(state: ChessGameState, depth: int, elapsed_time: float):
            if depth >= self.max_depth:
                return True
            else:
                return False

        best_move = heuristic_alphabeta_search(ChessProblem(board),ChessGameState(board),self.eval_function,cutoff_test)

        return best_move
