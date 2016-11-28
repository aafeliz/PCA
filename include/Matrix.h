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
class Matrix {
private:
    double* m;
    uint32_t rows,cols;
public:
    Matrix(uint32_t, uint32_t, double val = 0);
    ~Matrix();
    Matrix(const Matrix&);
    Matrix(Matrix&& );
    Matrix& operator =(const Matrix&);
    // retrieve matrix values from array requested in matrix from
    friend Matrix operator ()(int, int);


    Matrix operator T(const Matrix&);
    Matrix operator T(Matrix&);
    double operator T&(int, int); //T&
    double operator T(int, int); //T

    friend Matrix operator +(const Matrix&, const Matrix&);
    friend Matrix operator -(const Matrix&, const Matrix&);
    friend Matrix operator *(const Matrix&, const Matrix&);

    // add another matrix to this one, changing this
    Matrix operator +=(const Matrix&);
    // subtract another matrix from this one, changing this
    Matrix operator -=(const Matrix&);
    // as features with input data are being read the vectors get
    // placed into the matrix
    friend Matrix operator <<(const Matrix&);


    vector<double> gaussPartialPivoting(vector<double>&); // solve (*this)x = B, returning x.

    void gaussPartialPivoting(vector<double>&, vector<double>&); // solve (*this)x = B, modifying x that is passed by reference

    vector<double> gaussFullPivoting(vector<double>&); // solve (*this)x = B, returning x.

    void gaussFullPivoting(vector<double>&, vector<double>&); // solve (*this)x = B, modifying x that is passed by reference
    // a to the integer power k
    friend Matrix operator ^(const Matrix&, int);
    // write out matrix to a stream
    friend ostream& operator <<(ostream&, const Matrix&);
    // read in matrix from a stream
    friend istream& operator >>(istream&, Matrix&);

};
#endif //PCA_MATRIX_H
