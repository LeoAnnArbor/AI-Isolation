"""
test your agent's strength against a set of known agents using tournament.py

"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

# Helper Functions for Evaluators
def is_near_walls(move, walls):
    """
    Checks if a move is on  the edges of the board
    Parameters
    ----------
    move : (int, int)
        The input move on the board
    walls : list(list(tuples))
        A nested list of tuples for each edge of the board
    Returns
    -------
    bool
        Returns True if a move lies along the edges else False
    """
    for wall in walls:
        if move in wall:
            return True
    return False

def is_in_corners(move, corners):
    """
        Checks if a move is in the corners of the board
        Parameters
        ----------
        move : (int, int)
            The input move on the board
        corners : list(tuples)
            A list of tuples for each corner of the board
        Returns
        -------
        bool
            Returns True if a move lies in a corner of the board else False
    """
    return move in corners

def percent_occupied(game):
    """
            Parameters
            ----------
            game : `isolation.Board`
                The game board
            Returns
            -------
            int
                The percentage of occupied space in the board
        """
    blank_spaces = game.get_blank_spaces()
    return int((len(blank_spaces)/(game.width * game.height)) * 100)

def check_near_walls(game, player):
    """ 
    The  evaluation function calculates a cumulative score based on the moves and their positions.
    A cumulative score is calculated for both the players.
        
    Parameters
    ----------
    game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
    player : hashable
            One of the objects registered by the game object as a valid player.
            (i.e., `player` should be either game.__player_1__ or
            game.__player_2__).

    Returns
    ----------
    float
            The heuristic value of the current game state
    """
    walls = [
        [(0, i) for i in range(game.width)],
        [(i, 0) for i in range(game.height)],
        [(game.width - 1, i) for i in range(game.width)],
        [(i, game.height - 1) for i in range(game.height)]
    ]

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    own_cum_score = 0
    opp_cum_score = 0

    own_moves_left = 0
    opp_moves_left = 0

    own_moves_left = 0
    opp_moves_left = 0

    for move in own_moves:
        if is_near_walls(move, walls) and percent_occupied(game) < 50:
            own_cum_score += 10
        elif is_near_walls(move, walls) and percent_occupied(game) > 50 and percent_occupied(game) < 85:
            own_cum_score -= 20
        elif is_near_walls(move, walls) and percent_occupied(game) > 85:
            own_cum_score -= 30
        else:
            own_moves_left += 5

    for move in opp_moves:
        if is_near_walls(move, walls) and percent_occupied(game) < 50:
            opp_cum_score += 10
        elif is_near_walls(move, walls) and percent_occupied(game) > 50 and percent_occupied(game) < 85:
            opp_cum_score -= 20
        elif is_near_walls(move, walls) and percent_occupied(game) > 85:
            opp_cum_score -= 30
        else:
            opp_moves_left += 5

    return float(own_cum_score - opp_cum_score) + float(own_moves_left - opp_moves_left)

def check_in_corners(game, player):
    """ 
    The  evaluation function calculates a cumulative score based on the moves and their positions.
    A cumulative score is calculated for both the players.
        
    Parameters
    ----------
    game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
    player : hashable
            One of the objects registered by the game object as a valid player.
            (i.e., `player` should be either game.__player_1__ or
            game.__player_2__).

    Returns
    ----------
    float
            The heuristic value of the current game state
    """

    corners = [(0,0), (0,game.width-1), (game.height-1,0), (game.height-1,game.width-1)]

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    own_cum_score = 0
    opp_cum_score = 0
    own_moves_left = 0
    opp_moves_left = 0
    for move in own_moves:
        if is_in_corners(move, corners) and percent_occupied(game) < 60:
            own_cum_score += 15
        elif is_in_corners(move, corners) and percent_occupied(game) > 60:
            own_cum_score -= 40
        else:
            own_moves_left += 10

    for move in opp_moves:
        if is_in_corners(move, corners) and percent_occupied(game) < 60:
            opp_cum_score += 15
        elif is_in_corners(move, corners) and percent_occupied(game) > 60:
            opp_cum_score -= 40
        else:
            opp_moves_left += 10

    return float(own_cum_score - opp_cum_score) + float(own_moves_left - opp_moves_left)

def longest_moves(game, player, moves) :
    """Calculate the longest moves possible of a given player.
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    moves : list
        List of available legal moves for the player

    Returns
    -------
        
    float
        Length of longest moves available for the given player
    """
    longest_length = 0;
    length = 0;
    # basecase
    if len(moves) == 1:
        return len(moves)

    for move in moves:
        next_state = game.forecast_move(move)
        new_moves = next_state.get_legal_moves(player)

        length = longest_moves(next_state, player, new_moves) + 1

        if length > longest_length:
            longest_length = length
        
        # fail safe in case this heuristic is used too early
        if longest_length > 20: 
            break

    return longest_length   

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    This should be the best heuristic function for your project submission.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # own_moves = len(game.get_legal_moves(player))
    # opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # return float(own_moves - opp_moves) 

    # number of total possible moves
    total_moves = game.width * game.height
    # number of possible legal moves left
    moves_left = total_moves - game.move_count

    
    # First third of the game is considered as beginning stage
    #     - Number of moves player can make minus number of moves opponent can make
    #     - This provides a rough approxiamation of the "goodness" of a board,
    #       but it is fast and enables a deeper search
    #     - I chose search depth over heuristic quality for the beginning of the
    #       game because the board is very open 
    
    if moves_left > int(total_moves * 2 / 3):

        own_moves = len(game.get_legal_moves(player))
        opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

        return float(own_moves - opp_moves) 

    # Middle stage
    #     - Number of empty spaces in reachable area around the player minus number 
    #       of empty spaces in reachable area around opponent
    #     - This is a much better heuristic than NumberOfMoves, and helps to find
    #       situations where opponents are completely walled off from each other
    #     - This is slower than NumberOfMoves, but because it is only used in middle
    #       game, the board has fewer empty spaces, so it is not terrible
    
    elif moves_left <= int(total_moves * 2 / 3) and moves_left > int(total_moves / 3):
        
        walls_score = check_near_walls(game, player)
        corners_score = check_in_corners(game, player)
    
        return float(0.3 * walls_score + 0.7 * corners_score)

        # own_pos_y, own_pos_x = game.get_player_location(player)
        # opp_pos_y, opp_pos_x = game.get_player_location(game.get_opponent(player))

        # own_moves = game.get_legal_moves(player)
        # opp_moves = game.get_legal_moves(game.get_opponent(player))
       
        # # complete a list of adjacent area
        # own_adjacent = [(own_pos_y + 1, own_pos_x), (own_pos_y - 1, own_pos_x), (own_pos_y + 1, own_pos_x + 1),
        #                 (own_pos_y - 1, own_pos_x + 1), (own_pos_y + 1, own_pos_x - 1), (own_pos_y - 1, own_pos_x - 1), 
        #                 (own_pos_y , own_pos_x + 1), (own_pos_y, own_pos_x - 1)]

        # opp_adjacent = [(opp_pos_y + 1, opp_pos_x), (opp_pos_y - 1, opp_pos_x), (opp_pos_y + 1, opp_pos_x + 1),
        #                 (opp_pos_y - 1, opp_pos_x + 1), (opp_pos_y + 1, opp_pos_x - 1), (opp_pos_y - 1, opp_pos_x - 1), 
        #                 (opp_pos_y , opp_pos_x + 1), (opp_pos_y, opp_pos_x - 1)]

        # # check if the area around the player are legal moves
        # own_area = []
        # opp_area = []

        # for idx in range(len(own_adjacent)):
        #     if own_adjacent[idx] in own_moves:
        #         own_area.append(own_adjacent[idx])

        # for idx in range(len(opp_adjacent)):
        #     if opp_adjacent[idx] in opp_moves:
        #         opp_area.append(opp_adjacent[idx])

        # return float(len(own_area) - len(opp_area)) 
    
    # Final stage:
    #     - Calculate max number of moves player could make if the board froze right now, 
    #       assume other player doesn't move
    #     - Length of longest path player can walk minus length of longest path opponent can walk
    #     - This heuristic is perfectly accurate if the players are walled off from
    #       another, because whoever can make more moves will win
    #     - This is an expensive heuristic that would fail for any reasonable depth
    #       in the beginning or middle game
    
    else:
        own_moves = game.get_legal_moves(player)
        opp_moves = game.get_legal_moves(game.get_opponent(player))

        own_longest_moves = longest_moves(game, player, own_moves)
        opp_longest_moves = longest_moves(game, game.get_opponent(player), opp_moves)

        return float(own_longest_moves - opp_longest_moves)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(own_moves - 2* opp_moves) 


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

   
    own_location = game.get_player_location(player)
    opp_location = game.get_player_location(game.get_opponent(player))
    distance = abs(own_location[0]-opp_location[0]) + abs(own_location[1]-opp_location[1])

    return float(distance)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.
    ********************  DO NOT MODIFY THIS CLASS  ********************
    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************
        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.
        Parameters
        ----------
        game : `isolation.Board`

            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable

            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

            # return best_move

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

   

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm 
        
        Parameters
        ----------
        game : isolation.Board

            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int

            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------

        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        """
        # The helper functions below is a modified version of MINIMAX-DECISION in the AIMA text.
        # https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        def max_value(game, current_depth):

            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            legal_moves = game.get_legal_moves()

            if not legal_moves:
                return game.utility(self)

            if current_depth == 0:
                return self.score(game, self)

            best_score = float("-inf")

            for move in legal_moves:

                next_state = game.forecast_move(move)
                score = min_value(next_state, current_depth - 1)

                best_score = max(score, best_score)

            return best_score
 
        def min_value(game, current_depth):

            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            legal_moves = game.get_legal_moves()

            if not legal_moves:
                return game.utility(self)

            if current_depth == 0:
                return self.score(game, self)

            best_score = float("inf")

            for move in legal_moves:

                next_state = game.forecast_move(move)
                score = max_value(next_state, current_depth - 1)

                best_score = min(score, best_score)

            return best_score 

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()

        if not legal_moves or depth == 0:
            return (-1, -1)

        best_score = float("-inf")
        best_move = legal_moves[0]

        for move in legal_moves:
            next_state = game.forecast_move(move)
            score = min_value(next_state, depth - 1)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move              

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Implement iterative deepening search 
        legal_moves = game.get_legal_moves()

        # Are there any legal moves left for us to play? If not, we stop playing!
        if not legal_moves:
            return (-1, -1)

        # Did we just start the game? Then, pick the center position.
        if game.move_count == 0:
            
            return(int(game.height / 2), int(game.width / 2))

        # Let's search for a good move!
        best_move = (-1, -1)

        # Perform iterative deepening search
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            
            depth = 1

            while True:

                best_move = self.alphabeta(game, depth)

                if self.time_left() < 0:
                    break
                depth += 1
        
        except SearchTimeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning 
  
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        Returns
        -------
        float
            The score for the current search branch

        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        """
        
        # The helper functions below is a modified version of MINIMAX-DECISION in the AIMA text.
        # https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        def max_value(game, current_depth, alpha, beta):

            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            legal_moves = game.get_legal_moves()

            if not legal_moves:
                return game.utility(self)

            if current_depth == 0:
                return self.score(game, self)

            best_score = float("-inf")

            for move in legal_moves:

                next_state = game.forecast_move(move)
                score = min_value(next_state, current_depth - 1, alpha, beta)
                best_score = max(score, best_score)

                if best_score >= beta:
                    return best_score

                alpha = max(alpha, best_score)

            return best_score
 
        def min_value(game, current_depth, alpha, beta):

            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            legal_moves = game.get_legal_moves()

            if not legal_moves:
                return game.utility(self)

            if current_depth == 0:
                return self.score(game, self)

            best_score = float("inf")

            for move in legal_moves:

                next_state = game.forecast_move(move)
                score = max_value(next_state, current_depth - 1, alpha, beta)

                best_score = min(score, best_score)

                if best_score <= alpha:
                    return best_score

                beta = min(beta, best_score)

            return best_score 


        # Alpha is the maximum lower bound of possible solutions
        # Alpha is the highest score so far ("worst" highest score is -inf)
        
        # Beta is the minimum upper bound of possible solutions
        # Beta is the lowest score so far ("worst" lowest score is +inf)

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()

        if not legal_moves or depth == 0:
            return (-1, -1)

        best_score = float("-inf")
        best_move = legal_moves[0]
        alpha = float("-inf")
        beta = float("inf")

        for move in legal_moves:
            next_state = game.forecast_move(move)
            score = min_value(next_state, depth - 1, alpha, beta)
            
            if score > best_score:
                best_score = score
                best_move = move

            if best_score >= beta:
                return best_move

            alpha = max(alpha, best_score)

        return best_move   