#include "BackgammonUtils.h"
#include "Board.h"

#include <iostream>
#include <iomanip>

#include <windows.h>

using namespace BackgammonUtils;
using namespace std;
using namespace BoardBO;

namespace BackgammonUtils {
void print_board(const Board& board) {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    int s = board.size();
    for (size_t i = 0; i < 12; ++i) {
        if (board[i] > 0){
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_INTENSITY);
            std::cout << std::setw(3) << board[i];
        } else if (board[i] < 0){
            SetConsoleTextAttribute(hConsole, FOREGROUND_BLUE | FOREGROUND_INTENSITY);
            std::cout << std::setw(3) << board[i];
        } else {
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
            std::cout << std::setw(3) << " ";
        }
        if (i != board.size()/2 - 1) {
            std::cout << " "; // space between numbers
        }
    }
    std::cout << "\n";
    for (size_t i = 23; i >= 12; --i) {
        if (board[i] > 0){
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_INTENSITY);
            std::cout << std::setw(3) << board[i];
        } else if (board[i] < 0){
            SetConsoleTextAttribute(hConsole, FOREGROUND_BLUE | FOREGROUND_INTENSITY);
            std::cout << std::setw(3) << board[i];
        } else {
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
            std::cout << std::setw(3) << " ";
        }
        if (i != board.size() - 1) {
            std::cout << " "; // space between numbers
        }
    }
    std::cout << "\n";
    for (size_t i = 24; i < 28; ++i) {
        if (i == 24){
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
            cout<<"Bar: ";
        }
        if (i == 26){
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
            cout<<"             Off: ";
        }
        if (i == P1BAR || i == P1OFF){
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_INTENSITY);
            std::cout << std::setw(3) << board[i];
        } else {
            SetConsoleTextAttribute(hConsole, FOREGROUND_BLUE | FOREGROUND_INTENSITY);
            std::cout << std::setw(3) << board[i];
        }
        if (i != board.size() - 1) {
            std::cout << " "; // space between numbers
        }
    }

    SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    std::cout << std::endl;
}

void print_move_sequence(const MoveSequence& vec) {
    for (const auto& t : vec) {
        // Use unary + to print int8_t as number instead of char
        cout << "(" 
                  << +get<0>(t) << ", "
                  << +get<1>(t) << ", "
                  << +get<2>(t) << ") ";
    }
    cout << endl;
}

std::string move_sequence_to_string(const MoveSequence& moves) {
    if (moves.size() > 4) {
        throw std::runtime_error("MoveSequence cannot have more than 4 moves");
    }

    std::string result;
    for (size_t i = 0; i < 4; ++i) {
        if (i < moves.size()) {
            const Move& m = moves[i];
            result += std::to_string(std::get<0>(m)) + "." +
                      std::to_string(std::get<1>(m)) + "." +
                      std::to_string(std::get<2>(m));
        }
        if (i != 3) result += ",";
    }

    return result;
}

Board string_to_board(const string& input) {
    Board board{};
    stringstream ss(input);
    string token;
    size_t index = 0;

    while (getline(ss, token, ',')) {
        if (index >= board.size()) {
            throw runtime_error("Too many numbers in input string");
        }

        int num;
        try {
            num = stoi(token);
        } catch (...) {
            throw runtime_error("Invalid number: " + token);
        }

        if (num < -128 || num > 127) {
            throw out_of_range("Number out of int8_t range: " + token);
        }

        board[index++] = static_cast<int16_t>(num);
    }

    if (index != board.size()) {
        throw runtime_error("Too few numbers in input string");
    }

    return board;
}

Dice string_to_dice(const string& input) {
    Dice dice{};
    stringstream ss(input);
    string token;
    size_t index = 0;

    while (getline(ss, token, ',')) {
        if (index >= dice.size()) {
            throw runtime_error("Too many numbers in input string");
        }

        int num;
        try {
            num = stoi(token);
        } catch (...) {
            throw runtime_error("Invalid number: " + token);
        }

        if (num < -128 || num > 127) {
            throw out_of_range("Number out of int8_t range: " + token);
        }

        dice[index++] = static_cast<int8_t>(num);
    }

    if (index != dice.size()) {
        throw runtime_error("Too few numbers in input string");
    }

    return dice;
}

MoveSequence string_to_move_sequence(const string& input) {
    MoveSequence move_sequence;
    std::stringstream ss(input);
    std::string group;

    while (std::getline(ss, group, ',')) {
        if (group.empty()) continue; // skip empty groups

        std::stringstream gs(group);
        std::string token;
        int values[3];
        size_t i = 0;

        while (std::getline(gs, token, '.')) {
            if (i >= 3) throw std::runtime_error("Too many numbers in a group: " + group);
            
            int num;
            try {
                num = std::stoi(token);
            } catch (...) {
                throw std::runtime_error("Invalid number: " + token);
            }

            if (num < -128 || num > 127) {
                throw std::out_of_range("Number out of int8_t range: " + token);
            }

            values[i++] = static_cast<int8_t>(num);
        }

        if (i != 3) throw std::runtime_error("Too few numbers in a group: " + group);

        move_sequence.emplace_back(values[0], values[1], values[2]);
    }

    return move_sequence;
}
}







void t(){
// vector<BackgammonUtils::Move> BackgammonUtils::get_legal_moves(const vector<int>& board, int die, int player) {
//     vector<Move> moves;

//     if (board[Board::P1OFF] == 15 || board[Board::P2OFF] == -15)
//         return moves;

//     if (player == -1) {
//         if (board[Board::P2BAR] < 0) {
//             int end_pip = 24 - die;
//             if (board[end_pip] < 2)
//                 moves.push_back({Board::P2BAR, end_pip, die});
//         } else {
//             bool bearingOff = true;
//             for (int i = 6; i < 24; i++) if (board[i] < 0) { bearingOff = false; break; }

//             if (bearingOff) {
//                 int start_pip = die - 1;
//                 if (board[start_pip] < 0)
//                     moves.push_back({start_pip, Board::P2OFF, die});
//                 else {
//                     int furthest = -1;
//                     for (int i = 0; i < 7; i++) if (board[i] < 0) furthest = i;
//                     if (furthest != -1 && furthest < die)
//                         moves.push_back({furthest, Board::P2OFF, die});
//                 }
//             }

//             for (int start = 0; start < 24; start++) {
//                 if (board[start] < 0) {
//                     int end = start - die;
//                     if (end >= 0 && end < 24 && board[end] < 2)
//                         moves.push_back({start, end, die});
//                 }
//             }
//         }
//     } 
//     else if (player == 1) {
//         if (board[Board::P1BAR] > 0) {
//             int end_pip = die - 1;
//             if (board[end_pip] > -2)
//                 moves.push_back({Board::P1BAR, end_pip, die});
//         } else {
//             bool bearingOff = true;
//             for (int i = 0; i < 18; i++) if (board[i] > 0) { bearingOff = false; break; }

//             if (bearingOff) {
//                 int start_pip = 24 - die;
//                 if (board[start_pip] > 0)
//                     moves.push_back({start_pip, Board::P1OFF, die});
//                 else {
//                     int furthestIdx = -1;
//                     for (int i = 23; i >= 18; i--)
//                         if (board[i] > 0) { furthestIdx = i; break; }
//                     if (furthestIdx != -1 && furthestIdx > 24 - die)
//                         moves.push_back({furthestIdx, Board::P1OFF, die});
//                 }
//             }

//             for (int start = 0; start < 24; start++) {
//                 if (board[start] > 0) {
//                     int end = start + die;
//                     if (end >= 0 && end < 24 && board[end] > -2)
//                         moves.push_back({start, end, die});
//                 }
//             }
//         }
//     }

//     return moves;
// }

// // --- helper ---
// static vector<int> get_board_and_legal(vector<int> board, const BackgammonUtils::Move& m, int player) {
//     BackgammonUtils::do_next_board_partial(board, m, player);
//     return board;
// }

// // --- get_legal_move_sequences ---
// vector<BackgammonUtils::MoveSequence> BackgammonUtils::get_legal_move_sequences(const vector<int>& board, vector<int> dice, int player) {
//     set<vector<Move>> moveSequenceSet;
//     vector<MoveSequence> moveSequenceList;
//     sort(dice.begin(), dice.end());

//     auto add_unique = [&](const vector<Move>& seq) {
//         if (!moveSequenceSet.count(seq)) {
//             moveSequenceSet.insert(seq);
//             moveSequenceList.push_back(seq);
//         }
//     };

//     if (dice[0] == dice[1]) {
//         auto firstMoves = get_legal_moves(board, dice[0], player);
//         if (firstMoves.empty()) { add_unique({}); return moveSequenceList; }

//         for (auto m1 : firstMoves) {
//             auto b1 = get_board_and_legal(board, m1, player);
//             auto second = get_legal_moves(b1, dice[0], player);
//             if (second.empty()) { add_unique({m1}); continue; }
//             for (auto m2 : second) {
//                 auto b2 = get_board_and_legal(b1, m2, player);
//                 auto third = get_legal_moves(b2, dice[0], player);
//                 if (third.empty()) { add_unique({m1, m2}); continue; }
//                 for (auto m3 : third) {
//                     auto b3 = get_board_and_legal(b2, m3, player);
//                     auto fourth = get_legal_moves(b3, dice[0], player);
//                     if (fourth.empty()) { add_unique({m1, m2, m3}); continue; }
//                     for (auto m4 : fourth)
//                         add_unique({m1, m2, m3, m4});
//                 }
//             }
//         }
//     } else {
//         // dice order 1-2
//         for (int pass = 0; pass < 2; pass++) {
//             vector<int> d = dice;
//             if (pass == 1) swap(d[0], d[1]);
//             auto firstMoves = get_legal_moves(board, d[0], player);
//             for (auto m1 : firstMoves) {
//                 auto b1 = get_board_and_legal(board, m1, player);
//                 auto secondMoves = get_legal_moves(b1, d[1], player);
//                 if (secondMoves.empty()) add_unique({m1});
//                 else for (auto m2 : secondMoves) add_unique({m1, m2});
//             }
//         }
//     }

//     if (moveSequenceList.empty()) add_unique({});
//     return moveSequenceList;
// }

// void BackgammonUtils::do_next_board_partial(vector<int>& board, const Move& move, int player) {
//     if (move == Move{}) return;
//     auto [start, end, die] = move;

//     if (player == 1) {
//         if (board[end] == -1) {
//             board[end] = 0;
//             board[Board::P2BAR] -= 1;
//         }
//         board[start] -= 1;
//         board[end] += 1;
//     } else {
//         if (board[end] == 1) {
//             board[end] = 0;
//             board[Board::P1BAR] += 1;
//         }
//         board[start] += 1;
//         board[end] -= 1;
//     }
// }

// void BackgammonUtils::do_next_board_total(vector<int>& board, const MoveSequence& seq, int player) {
//     for (auto& move : seq) do_next_board_partial(board, move, player);
// }

// bool BackgammonUtils::game_over(const vector<int>& board) {
//     return (board[Board::P1OFF] == 15 || board[Board::P2OFF] == -15);
// }

// int BackgammonUtils::has_won(const vector<int>& board) {
//     if (board[Board::P1OFF] == 15) return 1;
//     if (board[Board::P2OFF] == -15) return -1;
//     return 0;
// }

// double BackgammonUtils::heuristic_evaluation(const vector<int>& board, int player) {
//     auto pip_count = [&](const vector<int>& b, int p) {
//         int p1 = 0, p2 = 0;
//         for (int i = 0; i < 24; i++) {
//             int n = b[i];
//             if (n > 0) p1 += n * (23 - i);
//             else if (n < 0) p2 += (-n) * i;
//         }
//         p1 += b[Board::P1BAR] * 24;
//         p2 += (-b[Board::P2BAR]) * 24;
//         return (p == 1) ? (p2 - p1) : (p1 - p2);
//     };

//     auto blot = [&](const vector<int>& b, int p) {
//         int s = 0;
//         for (int i = 0; i < 24; i++) {
//             if (b[i] == p) s -= 1;
//             else if (b[i] == -p) s += 1;
//         }
//         return s;
//     };

//     auto anchor = [&](const vector<int>& b, int p) {
//         int s = 0;
//         int start = (p == 1 ? 0 : 18);
//         int end = (p == 1 ? 6 : 24);
//         for (int i = start; i < end; i++) {
//             if (b[i] * p >= 2) s += 1;
//             else if (b[i] * p <= -2) s -= 1;
//         }
//         return s;
//     };

//     auto primes = [&](const vector<int>& b, int p) {
//         int self = 0, opp = 0, runSelf = 0, runOpp = 0;
//         for (int i = 0; i < 24; i++) {
//             if (b[i] * -p >= 2) { runOpp++; opp = max(opp, runOpp); }
//             else runOpp = 0;
//             if (b[i] * p >= 2) { runSelf++; self = max(self, runSelf); }
//             else runSelf = 0;
//         }
//         return self - opp;
//     };

//     auto bar = [&](const vector<int>& b, int p) {
//         int s = 0;
//         s -= (p == 1 ? b[Board::P1BAR] : -b[Board::P2BAR]);
//         s += (p == 1 ? -b[Board::P2BAR] : b[Board::P1BAR]);
//         return s;
//     };

//     auto off = [&](const vector<int>& b, int p) {
//         int s = 0;
//         s += (p == 1 ? b[Board::P1OFF] : -b[Board::P2OFF]);
//         s -= (p == 1 ? -b[Board::P2OFF] : b[Board::P1OFF]);
//         return s;
//     };

//     double Pi = 0.1 * pip_count(board, player);
//     double Bl = 1.5 * blot(board, player);
//     double An = 2.0 * anchor(board, player);
//     double Pr = 3.0 * primes(board, player);
//     double Ba = 7.0 * bar(board, player);
//     double Of = 5.0 * off(board, player);

//     return (Pi + Bl + An + Pr + Ba + Of) / 500.0;
// }
}