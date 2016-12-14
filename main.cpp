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

   
    //////////////////////////////////////////////////////
    //              USING JACOBIAN METHOD               //
    //////////////////////////////////////////////////////
    double *arr2 = new double[25];
    double input2[25] = {1,  7,  3,  7,  4,
                        -5,  3, -5,  6,  5,
                         2,  8,  5,  2,  3,
                         4,  5,  3,  3,  4,
                         4,  5,  6,  7,  3};
    
    for(size_t i = 0; i < 25; i++)
        arr2[i] = input2[i];
    
    Matrix mData2(5, 5, arr2);
    PCA p2(mData2);
    p2.outputData();
    p2.calcStats();
    p2.outputStats();
    p2.eigenJacobian();
    
    cout << "\n\nEig VALS:\n" << p2.eigenVals;
    cout << "\nEig VECS: \n" << p2.eigenVect;
    
    
    
    return 0;
}
