#include <iostream>
#include <vector>
#include <tuple>
#include ".\CPP\BackgammonUtils.h"

using namespace std;
using namespace BackgammonUtils;

void print_board(const vector<int>& board) {
    cout << "Board: ";
    for (int val : board) cout << val << " ";
    cout << endl;
}

void print_move(const Move& move) {
    int start, end, die;
    std::tie(start, end, die) = move;
    cout << "(" << start << " -> " << end << ", die=" << die << ")";
}

void print_move_sequence(const vector<Move>& sequence) {
    cout << "[ ";
    for (size_t i = 0; i < sequence.size(); ++i) {
        print_move(sequence[i]);
        if (i < sequence.size() - 1) cout << ", ";
    }
    cout << " ]" << endl;
}

int main() {
    vector<int> board = {2, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, -5, 0, 5, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0};
    vector<int> dice = {3, 4};

    cout << "Initial Board:" << endl;
    print_board(board);

    auto legal_moves = get_legal_moves(board, dice);
    cout << "\nLegal moves (count = " << legal_moves.size() << "):" << endl;
    for (size_t i = 0; i < legal_moves.size(); ++i) {
        cout << i + 1 << ": ";
        print_move(legal_moves[i]);
        cout << endl;
    }

    auto sequences = get_legal_move_sequences(board, dice);
    cout << "\nLegal move sequences (count = " << sequences.size() << "):" << endl;
    for (size_t i = 0; i < sequences.size(); ++i) {
        cout << i + 1 << ": ";
        print_move_sequence(sequences[i]);
    }

    return 0;
}
