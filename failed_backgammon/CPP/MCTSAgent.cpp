#include <iostream>
#include <cstdlib>
#include <cstring>
#include <windows.h>
#include "Board.h"
#include "BackgammonUtils.h"

using namespace std;
using namespace BackgammonUtils;


MoveSequence make_move(Board board, Dice dice, MoveSequence opp_move){
    ...
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: MCTSAgent <colour>" << endl;
        return 1;
    }

    int colour = stoi(argv[1]); // 1 = RED, -1 = BLUE

    string input_line;
    vector<string> parts;

    while (getline(cin, input_line)) {
        if (input_line.empty())
            continue;

        stringstream ss(input_line);
        string item;

        while (getline(ss, item, ';')) { // Use ';' as the delimiter
            parts.push_back(item);
        }

        if (parts.size() != 4){
            throw runtime_error("");
        }

        string command_str = parts[0];
        string board_str = parts[1];
        string dice_str = parts[2];
        string opp_move_str = parts[3];

        Board board = string_to_board(board_str);
        Dice dice = string_to_dice(dice_str);
        MoveSequence opp_move = string_to_move_sequence(opp_move_str);

        MoveSequence best_move = make_move(board,dice,opp_move);

        string s = move_sequence_to_string(opp_move);
        cout << s + "\n";
        cout.flush();
    }

    return 0;
}


// "12.17.5,17.23.6"
// START ;2,0,0,0,0,-5,0,-3,0,0,0,5,-5,0,0,0,3,0,5,0,0,0,0,-2,0,0,0,0;1,2;,,,
// CHANGE;2,0,0,0,0,-5,0,-3,0,0,0,4,-5,0,0,0,3,0,5,0,0,0,1,-2,0,0,0,0;4,5;12.17.5,17.23.6,,