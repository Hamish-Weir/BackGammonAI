#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    // Optional: read command-line args (e.g., colour, board size)
    // argv[1], argv[2], etc. if you need them later

    std::string line;

    // Read commands until stdin is closed
    while (std::getline(std::cin, line)) {
        // Expected format:
        // COMMAND;MOVESEQUENCE;BOARD;DICE;

        // You could parse here if needed, but for now do nothing

        // Return a blank response (just newline)
        std::cout << std::endl;  // endl flushes automatically
    }

    return 0;
}