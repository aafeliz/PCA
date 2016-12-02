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
#include "../include/Matrix.h"
#include <iostream>
#include <fstream>

/*
 * @todo: @bug redefinition issue with header when using it with : rows(rows)...
Matrix::Matrix(uint32_t rows, uint32_t cols, double val = 0) : rows(rows), cols(cols)
{
    m = new double[rows*cols];
    for (int i = 0; i < rows*cols; i++)
        m[i] = val;
}//*/
int Matrix::getIdx(int r, int c)
{
    return (rows*r + c);
}
int Matrix::getIdx(int r, int c) const
{
    return (rows*r + c);
}
Matrix::Matrix(uint32_t r, u_int32_t c,double *arr)
{
    rows = r, cols = c, m = arr;
    arr = nullptr;
}
Matrix::~Matrix()
{
    delete [] m;
}
/*
 *
 */
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
Matrix& Matrix::operator =(const Matrix& orig)
{
        if (this != &orig) {
            Matrix m(orig);
            return m;
        }
        return *this;
}

double Matrix::operator ()(int r, int c)
{
    return m[getIdx(r, c)];
}
double Matrix::operator()(int r, int c) const
{
    return m[Matrix::getIdx(r, c)];
}
/*
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
*/

Matrix operator +(const Matrix& a, const Matrix& b)
{
    if(a.rows != b.rows || a.cols != b.cols) exit(1);
    double *arr = new double[a.rows * a.cols];
    for(int r = 0; r < a.rows; r++)
    {
        for(int c = 0; c < a.cols; c++)
        {
            arr[((r*a.rows) + c)] = a(r, c) + b(r, c);
        }
    }
    // no need to delete arr since it becomes nullptr once Matrix is created
    return Matrix(a.rows, b.cols, arr);
}
Matrix operator -(const Matrix& a, const Matrix& b)
{
    if(a.rows != b.rows || a.cols != b.cols) exit(1);
    double *arr = new double[a.rows * a.cols];
    for(int r = 0; r < a.rows; r++)
    {
        for(int c = 0; c < a.cols; c++)
        {
            arr[((r*a.rows) + c)] = a(r, c) - b(r, c);
        }
    }
    // no need to delete arr since it becomes nullptr once Matrix is created
    return Matrix(a.rows, b.cols, arr);
}

Matrix operator *(const Matrix& a, const Matrix& b)
{
    if(a.cols != b.rows) exit(1);
    double *arr = new double[a.rows * b.cols];
    for(int ra = 0; ra < a.rows; ra++)
    {
        for(int cb = 0; cb < b.cols; cb++)
        {
            for(int ca = 0; ca < a.cols; ca++)
                arr[((ra * a.rows) + cb)] += a(ra, ca) * b(ca, cb);
        }
    }
    // no need to delete arr since it becomes nullptr once Matrix is created
    return Matrix(a.rows, b.cols, arr);
}

/*
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
*/
/*
// as features with input data are being read the vectors get
// placed into the matrix
friend Matrix::Matrix operator <<(const Matrix&)
{
    // @todo: implement function
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
*/
// a to the integer power k
/*friend Matrix& Matrix::operator ^(const Matrix& a, int k)
{
    // @todo: implement function
}*/

// write out matrix to a stream
std::ostream& operator<<(std::ostream& s, Matrix& m)
{
    for(int i = 0; i < m.rows; i++)
    {
        for(int j = 0; j < m.cols; j++)
        {
            s << m(i, j);
        }
        s << '\n';
    }
    return s;
}
/*
// read in matrix from a stream
friend Matrix::istream& operator >>(istream& s, Matrix& m)
{
    // @todo: implement function
}
*/
