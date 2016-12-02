#include <iostream>
#include "include/Matrix.h"

using namespace std;

int main() {
    Matrix m1(5, 5, 3);
    Matrix m2(5, 5, 4);
    Matrix m3 = m1 + m2;
    Matrix m4 = m3 - Matrix(5, 5, 7);
    Matrix m5 = Matrix(2, 2, 2) * Matrix(2, 2, 3);
    cout << "m1:\n" << m1 << endl;
    cout << "m2:\n" << m2 << endl;
    cout << "m3:\n" << m3 << endl;
    cout << "m4:\n" << m4 << endl;
    cout << "m5:\n" << m5 << endl;
    std::cout << "Hello, World!" << std::endl;
    return 0;
}