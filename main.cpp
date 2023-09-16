#include <iostream>
#include <fstream>
#include <string>

void copyFirstNLines(const std::string& sourceFileName, const std::string& destFileName, int n) {
    std::ifstream sourceFile(sourceFileName);
    if (!sourceFile.is_open()) {
        std::cerr << "Error: Unable to open source file." << std::endl;
        return;
    }

    std::ofstream destFile(destFileName);
    if (!destFile.is_open()) {
        std::cerr << "Error: Unable to open destination file." << std::endl;
        return;
    }

    std::string line;
    int linesCopied = 0;

    while (std::getline(sourceFile, line) && linesCopied < n) {
        destFile << line << std::endl;
        linesCopied++;
    }

    sourceFile.close();
    destFile.close();
}

int main() {
    std::string sourceFileName = "2020_d.csv";  // Change this to your source file's name
    std::string destFileName = "2020.csv";  // Change this to your destination file's name
    int n = 1000000;  // Number of lines to copy

    copyFirstNLines(sourceFileName, destFileName, n);

    std::cout << "Copied " << n << " lines from " << sourceFileName << " to " << destFileName << std::endl;

    return 0;
}
