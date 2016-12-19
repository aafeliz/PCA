#include <iostream>
#include "include/Matrix.h"
#include "include/PCA.h"
#include <cstdint>
using namespace std;
/*
The data to test PCA class comes from PCA hw for CPE::646
1	-3	4	-1	0	5	-1	3
2	-1	5	1	-2	2	-4	1
 */
int main()
{
    
    const size_t rows = 2,cols = 8;
    double in[rows][cols] = {{1, -3, 4, -1, 0, 5, -1, 3}, {2, -1, 5, 1, -2, 2, -4, 1}};
    double in2[rows*cols] = {1, -3, 4, -1, 0, 5, -1, 3, 2, -1, 5, 1, -2, 2, -4, 1};
    double **data;
    
    data = new double*[rows];
    for(size_t r = 0; r < rows; r++)
    {
        data[r] = new double[cols];
    }
    
    for(size_t r = 0; r < rows; r++)
    {
        for(size_t c = 0; c < cols; c++)
        {
            data[r][c] = in[r][c];
        }
    }
    
    double *data2 = new double[rows*cols];
    for(size_t i = 0; i< rows*cols; i++)
    {
        data2[i] = in2[i];
    }
    
    
    Matrix m(rows, cols, data);
    Matrix m2(rows, cols, data2);
    
    cout << m << endl;
    cout << m2 << endl;

    PCA lt(m);
    PCA lt2(m2);
    cout << "\n\n//////////////////////////////////////////////////////\n//              USING JACOBIAN METHOD LT             //\n//////////////////////////////////////////////////////\n" ;
    lt.outputData();
    lt2.outputData();
    lt.calcStats();
    lt.outputStats();
    /*
     mus should return :
     1
     0.5
     
     Scatter Matrix should return:
     54 37
     37 54
     
     */
    
    lt.calcEigen();
    lt.outputEigen();
    lt.calcPCA();
    lt.outputPCA();
    //1.7500   -1.7500    4.7500    0.2500   -0.7500    3.7500   -2.2500    2.2500
    //1.2500   -2.2500    4.2500   -0.2500   -1.2500    3.2500   -2.7500    1.7500
    
    cout << "\n\n//////////////////////////////////////////////////////\n//              USING JACOBIAN METHOD P2            //\n//////////////////////////////////////////////////////\n" ;
   
   
    double *arr2 = new double[25];
    double input2[25] = {1,  7,  3,  7,  4,
                        -5,  3, -5,  6,  5,
                         2,  8,  5,  2,  3,
                         4,  5,  3,  3,  4,
                         4,  5,  6,  7,  3};
    
    for(size_t i = 0; i < 25; i++)
        arr2[i] = input2[i];
    
    //Matrix mData2(5, 5, arr2);
    Matrix mData2;
    mData2.readFile("dataelv");
    PCA p2(mData2);
    p2.calcALL();
    p2.outputALL();
    //p2.outputData();
    //p2.calcStats();
    //p2.outputStats();
    //p2.calcEigen();
    //p2.outputEigen();
    //p2.calcPCA();
    //p2.outputPCA();
    
    cout << "\n\n//////////////////////////////////////////////////////\n//              USING JACOBIAN METHOD Elvin         //\n//////////////////////////////////////////////////////\n" ;
    cout << "\nElvin\n\n";
    Matrix elvin;
    elvin.readFile("/dataelv.csv");
    cout << '\n' << elvin;
    PCA elv(elvin);
    elv.outputData();
    elv.calcStats();
    elv.outputStats();
    elv.calcEigen();
    elv.outputEigen();
    elv.calcPCA();
    elv.outputPCA();
    
    cout << '\n' << elvin;
    
    
    
    return 0;
}
