/** @file Matrix.cpp
 *  @brief Principal Component Analysis Class.
 *
 *  Contains class to calculate most sinigicant feature for a given class
 *  and inputs for those features.
 *
 *  @author Ariel Feliz(aafeliz)
 *  @author Elvin (<githubid>)
 *  @author Dov Kruger
 *  @date 11/27/16
 *  @bug No known bugs.
 *  @todo implement empty methods
 */
#include <Matrix.h>
Matrix::Matrix(uint32_t rows, uint32_t cols, double val = 0) : rows(rows), cols(cols)
{
    m = new double[rows*cols];
    for (int i = 0; i < rows*cols; i++)
        m[i] = val;
}
Matrix::~Matrix()
{
    delete [] m;
}
Matrix::Matrix(const Matrix& orig) : rows(orig.rows), cols(orig.cols)
{
    m = new double[rows*cols];
    for (int i = 0; i < rows*cols; ++i)
        m[i] = orig.m[i];
}
Matrix::Matrix(Matrix&& orig) : rows(orig.rows), cols(orig.cols), m(orig.m)
{
    // might want to keep original
    //orig.m = nullptr;
}
Matrix& operator =(const Matrix& orig)
{
        if (this != &orig) {
            //TODO: copy goes here.  Try to use copy constructor in C++11, need to check!
        }
        return *this;
}

Matrix Matrix:: operator T&(int r, int c)
{
    // @todo: implement function
    Matrix mx;
    return mx;
}
Matrix Matrix:: operator T(int r, int c) const
{
    // @todo: implement function
    Matrix mx;
    return mx;
}
double Matrix:: operator T&(int r, int c)
{
    return m[r*cols + c];
}
double Matrix:: operator T(int r, int c) const
{
    return m[r*cols + c];
}

friend Matrix::Matrix operator +(const Matrix& a, const Matrix& b);
friend Matrix::Matrix operator -(const Matrix& a, const Matrix& b);
friend Matrix::Matrix operator *(const Matrix& a, const Matrix& b);

// add another matrix to this one, changing this
Matrix Matrix::operator +=(const Matrix& b)
{
    // @todo: implement function
    Matrix mx;
    return mx;
}
// subtract another matrix from this one, changing this
Matrix Matrix::operator -=(const Matrix& b)
{
    // @todo: implement function
    Matrix mx;
    return mx;
}
vector<double> Matrix::gaussPartialPivoting(vector<double>& B) // solve (*this)x = B, returning x.
{
    // @todo: implement function

    return vector<double>;
}
void Matrix::gaussPartialPivoting(vector<double>&x, vector<double>& B)// solve (*this)x = B, modifying x that is passed by reference
{
    // @todo: implement function
    return;
}

vector<double> Matrix::gaussFullPivoting(vector<double>& B) // solve (*this)x = B, returning x.
{
    // @todo: implement function
}
void Matrix::gaussFullPivoting(vector<double>&x, vector<double>& B) // solve (*this)x = B, modifying x that is passed by reference
{
    // @todo: implement function
}
// a to the integer power k
friend Matrix::Matrix operator ^(const Matrix& a, int k);
{
    // @todo: implement function
}

// write out matrix to a stream
friend Matrix::ostream& operator <<(ostream& s, const Matrix& m);
{
    // @todo: implement function
}

// read in matrix from a stream
friend Matrix::istream& operator >>(istream& s, Matrix& m);
{
    // @todo: implement function
}

