#include <iostream>
#include "Matrix.h"
#include "PCA.h"
#include <cstdint>
using namespace std;

int main()
{
    Matrix m1(5, 5, 2);
    cout << m1 << endl;
    PCA pca(m1);

    
    return 0;
}
