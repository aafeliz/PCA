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
    const size_t rows = 3,cols = 8;
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

    //Matrix me;
    //load matrix with csv file
    //me.readFile("data2.csv");
    //me.writeFile("matrix.csv");
    //cout << me;
    double *arr = new double[9];
    double input[] = {1, 0, 2, 0, 2, 0, 2, 0, 1};
    for(size_t i = 0; i < 9; i++)
    {
        arr[i] = input[i];
    }
    Matrix mData(3, 3, arr);
    PCA p(mData);
    p.outputData();
    p.calcStats();
    p.outputStats();
    //Matrix A = lt.getScatter();
    Matrix *eig2 = lt.eigen();
    cout << "Eig[0]:\n" << eig2[0];
    
    cout << "\nEig[1]: \n" << eig2[1];
    Matrix *eig = p.eigen();//new Matrix[2];
    
    //A.eigen(A, eig[0], eig[1]);
    
    cout << "Eig[0]:\n" << eig[0];
    
    cout << "\nEig[1]: \n" << eig[1];
    
    
    
    return 0;
}
