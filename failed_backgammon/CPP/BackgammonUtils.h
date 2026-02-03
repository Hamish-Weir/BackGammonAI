#pragma once
#include <vector>
#include <tuple>
#include <string>
#include <sstream>
#include <array>

namespace BackgammonUtils {
    using Move = std::tuple<int8_t,int8_t,int8_t>;
    using MoveSequence = std::vector<Move>;
    using Board = std::array<int16_t, 28>;
    using Dice = std::array<int8_t,2>;

    void print_board(const std::array<int16_t, 28>& board);
    void print_move_sequence(const MoveSequence& vec);
    std::string move_sequence_to_string(const MoveSequence& moves);

    Board string_to_board(const std::string& input);
    Dice string_to_dice(const std::string& input);
    MoveSequence string_to_move_sequence(const std::string& input);

}
// static std::vector<Move> get_legal_moves(const std::vector<int>& board, int die, int player);
// static std::vector<MoveSequence> get_legal_move_sequences(const std::vector<int>& board, std::vector<int> dice, int player);
// static void do_next_board_partial(std::vector<int>& board, const Move& move, int player);
// static void do_next_board_total(std::vector<int>& board, const MoveSequence& move_sequence, int player);
// static bool game_over(const std::vector<int>& board);
// static int has_won(const std::vector<int>& board);
// static double heuristic_evaluation(const std::vector<int>& board, int player);

