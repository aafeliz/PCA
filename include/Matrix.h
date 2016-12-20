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
#include <vector>// vector
#include <sstream>
#include <string>
#include <iomanip>



/**@todo: switch everything to template type to make class general to data type.
                Matrix operator T(const Matrix&);
                Matrix operator T(Matrix&);
                double operator T&(size_t, size_t); //T&
                double operator T(size_t, size_t); //T
 *        switch size type to size_t instead of int
 *
 */

class Matrix
{
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
    /**@brief: retrieve individual matrix values from array requested in matrix from
        a double
     */
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
    // scalar gain on a matrix
    friend Matrix operator *(const double& a, const Matrix& b);


    // add another matrix to this one, changing this
    Matrix operator +=(const Matrix&);
    // scalar and matrix addition
    Matrix operator +=(const double& b);
    // subtract another matrix from this one, changing this
    Matrix operator -=(const Matrix&);
    
    
    //double* thMaxElement();
    
    
    
    Matrix* jacobian_eig();
   
    
    // will return matrix to qrDecomp
    //Matrix houseHolder(const Matrix& A);

    // read CSV file and generate Matrix
	void readFile(std::string filename);
	void writeFile(std::string filename);

    // get a column or row in matrix form
    Matrix getColumn(size_t col);
    Matrix getRow(size_t row);
    Matrix unitVector(size_t col);
    
    void appendRow(const Matrix& B);
    void appendCol(const Matrix& B);
    
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

    void sortRow(size_t r);
    void switchColumn(size_t a, size_t b);
 

};
namespace Mat {
    size_t getMaxIdx(const Matrix&);
    size_t getIdx(const size_t& Rows, const size_t& Cols, const size_t& r, const size_t& c);
    void sortRowAB_BasedOnA(const size_t& r, Matrix& A, Matrix& B);
    /**@brief: Jacobian eigen decomposition helper function find max value
     * Will retrieve value and location of largest value of top half of matrix
     * Returns drouble array with [maxVal, Row, Col]
     */
    double* thMaxElement(const Matrix& A);
    Matrix* jacobian_eig(const Matrix& A);
    /**@brief: Jacobian eigen decomp helper function to rotate matrix*/
    void mRotate(Matrix& A, Matrix& T, const u_int32_t& R, const u_int32_t& C);
}
#endif //PCA_MATRIX_H
