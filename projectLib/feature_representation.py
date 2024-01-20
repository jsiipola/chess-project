#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 19:28:06 2022

@author: siipola
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 08:22:22 2022

@author: siipola
"""

import time
import chess
import chess.svg
import numpy as np
import random
import sys

"""
Aim of this script is to write something that handles the feature representation
of the chess board and game situation.

"""

class feature_representation(object):
    def __init__(self,id_):
        self.id = id_
        self.n_features = 75
        
#class featureRepresentation(object):
    
#    def __init__(self):
#        self.id = 0
        
    def display_board(self, board):
        display(board)
        time.sleep(1)
            
    def piece_position(self, key, position_dict, num):
        if self.piece_exists(key, position_dict, num):
            list_ = position_dict[key]
            return list_[num-1]
        else:
            return float("nan")
            
    def piece_exists(self, key, position_dict, num):
        if key not in position_dict:
            return False
        list_ = position_dict[key]
        n = len(list_)
        if n<num:
            return False
        return True
            
    def from_fen_to_piece_positions(self, fen):
        hash_map = {}
        row = 7
        col = -1
        for str_ in fen:
            col += 1
            if str_ == "/": # changing the row
                col = -1
                row = row -1
                continue
            elif str_.isnumeric():
                num = int(str_)
                col = col + num
                continue
            else:
                pos = row*8 + col # define the position of the piece
                """ Each piece has a key to the hashmap. position is a value in the
                    hash map. """
                if str_ in hash_map.keys():
                    hash_map[str_].append(pos)
                else:
                    hash_map[str_] = []
                    hash_map[str_].append(pos)
        return hash_map
        
    def play_turn(self, board):
        move_list = []
        for move in board.legal_moves:
            move_list.append(move)
    #    move_list = random.shuffle(move_list)
        k = board.legal_moves.count()
        x = np.random.rand()
        for i in range(k):
            i += 1
            if (float(i)/float(k)) > x:
                break
        move_index = i-1
        # after break i is the correct index.
    #    print(move_list)
        move = move_list[move_index]
        board.push(move)
        return board
    
    def test_if_legal_moves(self, board):
        g = board.legal_moves.count()
        if g>0:
            return True
        else:
            return False
    
    def detect_draws(self, board):
        """ Return a True if the game has run into draw. Otherwise returns a False. """
        is_draw = False
        if board.is_stalemate():
            is_draw = True
        elif board.is_insufficient_material():
            is_draw = True
        elif board.can_claim_threefold_repetition():
            is_draw = True
#        if board.halfmove_clock()
        elif board.can_claim_fifty_moves():
            is_draw = True
        elif board.can_claim_draw():
            is_draw = True
        elif board.is_fivefold_repetition():
            is_draw = True
        elif board.is_seventyfive_moves():
            is_draw = True
        return is_draw
    
    def detect_check_mate(self, board):
        return board.is_checkmate()
        
    def feature_representation(self, board):
        def num_of_pieces(piece):
            """
            Produces the feature representation of the chess board position.
            
            arg: piece = should be a string. Refer to a piece in the chess board.
            """
            if piece in position_dict.keys():
                pos_list = position_dict[piece]
                num_features = len(pos_list)
            else:
                num_features = 0
            return num_features
        
        position_dict = self.from_fen_to_piece_positions(board.board_fen())
        features = np.zeros((self.n_features,1))
        features[0] = board.turn # side to move
        features[1] = board.has_queenside_castling_rights(chess.WHITE) # white long castle
        features[2] = board.has_kingside_castling_rights(chess.WHITE) # white short castle
        features[3] = board.has_queenside_castling_rights(chess.BLACK)
        features[4] = board.has_kingside_castling_rights(chess.BLACK)
        
        features[5] = num_of_pieces('Q') # white queens
        features[6] = num_of_pieces('R') # white rooks
        features[7] = num_of_pieces('B') # white bishops
        features[8] = num_of_pieces('N') # white knights
        features[9] = num_of_pieces('P') # white pawns
        features[10] = num_of_pieces('q') # black queens
        features[11] = num_of_pieces('r') # black rooks
        features[12] = num_of_pieces('b') # black bishops
        features[13] = num_of_pieces('n') # black knights
        features[14] = num_of_pieces('p') # black pawns
        
        # White King position (This not used in Giraffe?)
        features[15] = board.king(chess.WHITE)
        # White queen1 exists
        features[16] = self.piece_exists('Q', position_dict, 1)
        # White queen1 position
        features[17] = self.piece_position('Q', position_dict, 1)
        # White Rook 1 Exists
        features[18] = self.piece_exists('R', position_dict, 1)
        # White Rook 1 Position
        features[19] = self.piece_position('R', position_dict, 1)
        # White Rook 2 Exists
        features[20] = self.piece_exists('R', position_dict, 2)
        # White Rook 2 Position
        features[21] = self.piece_position('R', position_dict, 2)
        # White Bishop 1 Exists
        features[22] = self.piece_exists('B', position_dict, 1)
        # White Bishop 1 Position
        features[23] = self.piece_position('B', position_dict, 1)
        # White Bishop 2 Exists
        features[24] = self.piece_exists('B', position_dict, 2)
        # White Bishop 2 Position
        features[25] = self.piece_position('B', position_dict, 2)
        # White Knight 1 Exists
        features[26] = self.piece_exists('N', position_dict, 1)
        # White Knight 1 Position
        features[27] = self.piece_position('N', position_dict, 1)
        # White Knight 2 Exists
        features[28] = self.piece_exists('N', position_dict, 2)
        # White Knight 2 Position
        features[29] = self.piece_position('N', position_dict, 2)
        # White pawn 1 Exists
        features[28] = self.piece_exists('P', position_dict, 1)
        # White pawn 1 Position
        features[29] = self.piece_position('P', position_dict, 1)
        # White pawn 2 Exists
        features[30] = self.piece_exists('P', position_dict, 2)
        # White pawn 2 Position
        features[31] = self.piece_position('P', position_dict, 2)
        # White pawn 3 Exists
        features[32] = self.piece_exists('P', position_dict, 3)
        # White pawn 3 Position
        features[33] = self.piece_position('P', position_dict, 3)
        # White pawn 4 Exists
        features[34] = self.piece_exists('P', position_dict, 4)
        # White pawn 4 Position
        features[35] = self.piece_position('P', position_dict, 4)
        # White pawn 5 Exists
        features[36] = self.piece_exists('P', position_dict, 5)
        # White pawn 5 Position
        features[37] = self.piece_position('P', position_dict, 5)
        # White pawn 6 Exists
        features[38] = self.piece_exists('P', position_dict, 6)
        # White pawn 6 Position
        features[39] = self.piece_position('P', position_dict, 6)
        # White pawn 7 Exists
        features[40] = self.piece_exists('P', position_dict, 7)
        # White pawn 7 Position
        features[41] = self.piece_position('P', position_dict, 7)
        # White pawn 8 Exists
        features[42] = self.piece_exists('P', position_dict, 8)
        # White pawn 8 Position
        features[43] = self.piece_position('P', position_dict, 8)
        
        # Black King position (This not used in Giraffe?)
        features[44] = board.king(chess.BLACK)
        # Black queen1 exists
        features[45] = self.piece_exists('q', position_dict, 1)
        # Black queen1 position
        features[46] = self.piece_position('q', position_dict, 1)
        # Black Rook 1 Exists
        features[47] = self.piece_exists('r', position_dict, 1)
        # Black Rook 1 Position
        features[48] = self.piece_position('r', position_dict, 1)
        # Black Rook 2 Exists
        features[49] = self.piece_exists('r', position_dict, 2)
        # Black Rook 2 Position
        features[50] = self.piece_position('r', position_dict, 2)
        # Black Bishop 1 Exists
        features[51] = self.piece_exists('b', position_dict, 1)
        # Black Bishop 1 Position
        features[52] = self.piece_position('b', position_dict, 1)
        # Black Bishop 2 Exists
        features[53] = self.piece_exists('b', position_dict, 2)
        # Black Bishop 2 Position
        features[54] = self.piece_position('b', position_dict, 2)
        # Black Knight 1 Exists
        features[55] = self.piece_exists('n', position_dict, 1)
        # Black Knight 1 Position
        features[56] = self.piece_position('n', position_dict, 1)
        # Black Knight 2 Exists
        features[57] = self.piece_exists('n', position_dict, 2)
        # Black Knight 2 Position
        features[58] = self.piece_position('n', position_dict, 2)
        # Black pawn 1 Exists
        features[59] = self.piece_exists('p', position_dict, 1)
        # Black pawn 1 Position
        features[60] = self.piece_position('p', position_dict, 1)
        # Black pawn 2 Exists
        features[61] = self.piece_exists('p', position_dict, 2)
        # Black pawn 2 Position
        features[62] = self.piece_position('p', position_dict, 2)
        # Black pawn 3 Exists
        features[63] = self.piece_exists('p', position_dict, 3)
        # Black pawn 3 Position
        features[64] = self.piece_position('p', position_dict, 3)
        # Black pawn 4 Exists
        features[65] = self.piece_exists('p', position_dict, 4)
        # Black pawn 4 Position
        features[66] = self.piece_position('p', position_dict, 4)
        # Black pawn 5 Exists
        features[67] = self.piece_exists('p', position_dict, 5)
        # Black pawn 5 Position
        features[68] = self.piece_position('p', position_dict, 5)
        # Black pawn 6 Exists
        features[69] = self.piece_exists('p', position_dict, 6)
        # Black pawn 6 Position
        features[70] = self.piece_position('p', position_dict, 6)
        # Black pawn 7 Exists
        features[71] = self.piece_exists('p', position_dict, 7)
        # Black pawn 7 Position
        features[72] = self.piece_position('p', position_dict, 7)
        # Black pawn 8 Exists
        features[73] = self.piece_exists('p', position_dict, 8)
        # Black pawn 8 Position
        features[74] = self.piece_position('p', position_dict, 8)
        
        
        
        
        
        return features
        #    features[4] # balck queens
        #    features[4] # black rooks
        #    features[4] # black bishops
        #    features[4] # black knights
        #    features[4] # black pawns
        #    features[4] # white queens exists
        #    features[4] # white queen 1 position
        #    features[4] # white rook 1 exists
        #    features[4] # white rook position
        #    features[4] # white rook 2 exists
            
            
def main():
    board = chess.Board()
    chess.svg.board(board, size=350)
    
    while True:
        if not test_if_legal_moves(board):
            break
        
        board = play_turn(board)
        print(board.turn) # side to move
        print(bool(board.castling_rights & chess.BB_A1))
        print(bool(board.castling_rights & chess.BB_H1))
        print(board.has_kingside_castling_rights(chess.WHITE))
        print(chess.BB_H1)
        display_board(board)
        print(board)
        print(board.piece_at(chess.A1))
#        print(board.R(chess.WHITE))
        print(board.board_fen())
        hash_map = from_fen_to_piece_positions(board.board_fen())
        print(hash_map)
        vec = feature_representation(board)
        print(vec)
        sys.exit()
    
    print(board)
    
    display_board(board)


if __name__ == "__main__":
    main()











