#include <iostream>
#include "include/Matrix.h"
#include <cstdint>
using namespace std;

int main() {
    Matrix m1(5, 5, 3);
    cout << "m1:\n" << m1 << endl;

    Matrix m2(5, 5, 4);
    cout << "m2:\n" << m2 << endl;

    Matrix m3 = m1 + m2;
    cout << "m3 sum:\n" << m3 << endl;

    Matrix mdif(5, 5, 7);
    Matrix m4 = m3 - mdif;//Matrix(5, 5, 7);
    cout << "m4 dif:\n" << m4 << endl;

    Matrix mult1(2,2,2);
    Matrix mult2(2,2,3);
    Matrix m5 = mult1 * mult2;//Matrix(2, 2, 2) * Matrix(2, 2, 3);
    cout << "m5 mult:\n" << m5 << endl;

    size_t rows, cols;
    cout << "enter num of ROWS: ";
    cin >> rows;
    cout << "\nenter num of COLUMNS: ";
    cin >> cols;
    double* m = new double[rows * cols];
    double val;
    for(int r = 0; r < rows; r++)
    {
        for(int c = 0; c < cols; c++)
        {
            //cout << "Enter value to go ino,  Row " << r+1 << "and Column " << c+1 << " : ";
            //cin >> val;
            //cout << '\n';
            val = r*cols + c;
            m[r*cols + c] = val+1;
        }
    }
    Matrix m6(rows, cols, m);






    cout << "m6:\n" << m6 << endl;
    m5 = m6;
    ~m6;
    cout << "~m6:\n" << m6 << endl;

    const Matrix con = m6;
    Matrix m7 = ~con;
    cout << "m7:\n" << m7 << endl;
    ~m7;
    cout << "~m7:\n" << m7 << endl;

    Matrix m8 = m1;
    Matrix m9 = m2;
    m8 += m9;
    cout << "m8 +=: \n" << m8 << endl;
    m9 -= m8;
    cout << "m9 -=: \n" << m9 << endl;
    cout << "Hello, World!" << endl;
    return 0;
}
