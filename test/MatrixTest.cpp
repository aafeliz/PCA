//
//  MatrixTest.cpp
//  PCA
//
//  Created by Ariel Feliz on 12/7/16.
//  Copyright © 2016 Ariel Feliz. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include "Matrix.h"
#include <cstdint>
using namespace std;

void TestM() {
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
    for(size_t r = 0; r < rows; r++)
    {
        for(size_t c = 0; c < cols; c++)
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
    
    Matrix v1(1,5,1);
    cout << "v1: \n" << v1 << endl;
    Matrix m10 = m1 - v1;
    cout << "m10 -: \n" << m10 << endl;
    
    double* tmp = new double[5]{1,2,3,4,5};
    
    Matrix v2(5,1,tmp);
    cout << "v2: \n" << v2 << endl;
    Matrix m11 = m1 - v2;
    cout << "m11 -: \n" << m11 << endl;
    
    Matrix pt1(1,1,5);
    cout << "pt1: \n" << pt1 << endl;
    Matrix v3 = v2 - pt1;
    cout << "v3 - 1 block: \n" << v3 << endl;
    
    //expected to exit program bc this is not allowed
    Matrix m13 = v1 - m1;
    cout << "test\n";
    return;
}
