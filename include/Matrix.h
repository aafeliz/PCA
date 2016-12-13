/** @file Matrix.cpp
 *  @brief Principal Component Analysis Class.
 *
 *  Contains class to calculate most sinigicant feature for a given class
 *  and inputs for those features.
 *
 *  @author Ariel Feliz(aafeliz)
 *  @author Dov Kruger
 *  @author Elvin Abreu(elvinabreu)
 *  @date 11/27/16
 *  @bug No known bugs.
 *  @todo Need to create methods for eigen decomposition. Providing PCA class with eigen value & vector.
 */

#ifndef PCA_MATRIX_H
#define PCA_MATRIX_H
#include <fstream> // ostream, ofstream
#include <iostream>
#include <stdlib.h>
#include <cmath> // sqrt, abs
/**@todo: switch everything to template type to make class general to data type.
                Matrix operator T(const Matrix&);
                Matrix operator T(Matrix&);
                double operator T&(size_t, size_t); //T&
                double operator T(size_t, size_t); //T
 *        switch size type to size_t instead of int
 *
 */
class Matrix {
private:
    double* m;
    size_t getIdx(size_t r, size_t c);
    size_t getIdx(size_t r, size_t c) const;
public:
    size_t rows, cols;
    Matrix():m(nullptr),rows(0),cols(0) {}
    
    /**@brief: constructor for recieving a double array in the form of double pointer
     * data[r][c]  
     */
    Matrix(size_t rows, size_t cols, double **data);
    
    Matrix(size_t rows, size_t cols, double val = 0): rows(rows), cols(cols)
    {
        m = new double[rows*cols];
        for (size_t i = 0; i < rows*cols; i++)
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
    double& operator ()(size_t r, size_t c);
    double& operator ()(size_t r, size_t c) const;
    //double& operator[]( size_t, size_t);
    inline void setValue(size_t, size_t, double);

    /**@brief: matrix transpose operators
     * T ===> ~
     */

    friend Matrix& operator ~(const Matrix&);
    /**@brief: transposes matrix being passed in
     */
    friend void operator ~(Matrix&);
    //double operator ~(size_t, size_t);


    friend Matrix operator +(const Matrix&, const Matrix&);
    friend Matrix operator -(const Matrix&, const Matrix&);
    friend Matrix operator *(const Matrix&, const Matrix&);


    // add another matrix to this one, changing this
    Matrix operator +=(const Matrix&);
    // subtract another matrix from this one, changing this
    Matrix operator -=(const Matrix&);
    
    /**@todo:
     */
    // diganoal everything else zero
    Matrix eye(const Matrix& orig);
    Matrix minor(const Matrix& z, const Matrix& k);
    Matrix* eigen(const Matrix& A);
    void eigen(const Matrix& A, Matrix& Q, Matrix& R);
    
    void houseHolder(Matrix& A, Matrix& d, Matrix& e, const size_t& row);
    void ql(Matrix& A, Matrix& d, Matrix& e, const size_t& row);
    Matrix makeD(const Matrix& d, const Matrix& e);
    
    
    // will return array as [Q, R]
    Matrix* qrDecomp(const Matrix& A);
    // will change Q and R passed in
    void qrDecomp(const Matrix& A, Matrix& Q, Matrix& R);
    
    // will return matrix to qrDecomp
    //Matrix houseHolder(const Matrix& A);

    // get a column or row in matrix form
    Matrix getColumn(size_t col);
    Matrix getRow(size_t row);
    Matrix unitVector(size_t col);
    
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
    //friend Matrix& operator ^(const Matrix&, size_t);
    /**@brief: write I/O matrix to a stream
     */
    
    
    friend std::ostream& operator<<(std::ostream&, const Matrix&); // no const because using non const type operator()
    // read in matrix from a stream
    friend std::istream& operator >>(std::istream&, Matrix&);

    inline void setRows(size_t);
    inline void setCols(size_t);

};
#endif //PCA_MATRIX_H
