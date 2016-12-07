/** @file Matrix.cpp
 *  @brief Principal Component Analysis Class.
 *
 *  Contains class to calculate most sinigicant feature for a given class
 *  and inputs for those features.
 *
 *  @author Ariel Feliz(aafeliz)
 *  @author Dov Kruger
 *  @author Elvin (<githubid>)
 *  @date 11/27/16
 *  @bug No known bugs.
 *  @todo Need to create methods for eigen decomposition. Providing PCA class with eigen value & vector.
 */

#ifndef PCA_MATRIX_H
#define PCA_MATRIX_H
#include <fstream> // ostream, ofstream
#include <iostream>
#include <stdlib.h>
/**@todo: switch everything to template type to make class general to data type.
                Matrix operator T(const Matrix&);
                Matrix operator T(Matrix&);
                double operator T&(int, int); //T&
                double operator T(int, int); //T
 *        switch size type to size_t instead of int
 *
 */
class Matrix {
private:
    double* m;
    int getIdx(int r, int c);
    int getIdx(int r, int c) const;
public:

    size_t rows,cols;
    Matrix(size_t rows, size_t cols, double val = 0): rows(rows), cols(cols)
    {
        m = new double[rows*cols];
        for (int i = 0; i < rows*cols; i++)
            m[i] = val;
    }
    // will be used to make matrix from array; take a look in how it is use in +, -, and *
    Matrix(size_t r, size_t c, double *);
    ~Matrix();

    /**@brief: c++ 11 copy constructor
     */
    Matrix(const Matrix&);
    Matrix(Matrix&& );
    Matrix& operator =(const Matrix&);
    // retrieve individual matrix values from array requested in matrix from
    double operator ()(int r, int c);
    double operator ()(int r, int c) const;
    //double& operator[]( int, int);
    inline double setValue(int, int, double);

    /**@brief: matrix transpose operators
     * T ===> ~
     */

    friend Matrix operator ~(const Matrix&);
    //friend Matrix operator ~(Matrix&);
    /**@brief: transposes matrix being passed in
     */
    friend void operator ~(Matrix&);
    //double operator ~(int, int);


    friend Matrix operator +(const Matrix&, const Matrix&);
    friend Matrix operator -(const Matrix&, const Matrix&);
    friend Matrix operator *(const Matrix&, const Matrix&);


    // add another matrix to this one, changing this
    Matrix operator +=(const Matrix&);
    // subtract another matrix from this one, changing this
    Matrix operator -=(const Matrix&);
    // subtract and 1xn or nx1 matrix from this matrix
    void minusVect(const Matrix& vec);


    /*
    // as features with input data are being read the vectors get
    // placed into the matrix
    friend std::ostream& operator <<(std::ostream, const Matrix&);


    vector<double> gaussPartialPivoting(vector<double>&); // solve (*this)x = B, returning x.

    void gaussPartialPivoting(vector<double>&, vector<double>&); // solve (*this)x = B, modifying x that is passed by reference

    vector<double> gaussFullPivoting(vector<double>&); // solve (*this)x = B, returning x.

    void gaussFullPivoting(vector<double>&, vector<double>&); // solve (*this)x = B, modifying x that is passed by reference
    */
    // a to the integer power k
    //friend Matrix& operator ^(const Matrix&, int);
    /**@brief: write I/O matrix to a stream
     */
    friend std::ostream& operator<<(std::ostream&, const Matrix&); // no const because using non const type operator()
    // read in matrix from a stream
    friend std::istream& operator >>(std::istream&, Matrix&);

    inline void setRows(size_t);
    inline void setCols(size_t);

};
#endif //PCA_MATRIX_H
