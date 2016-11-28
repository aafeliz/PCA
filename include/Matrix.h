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
    Matrix(uint32_t rows, uint32_t cols, double val = 0);
    ~Matrix();
    Matrix(const Matrix&);
    Matrix(Matrix&& );
    Matrix& operator =(const Matrix&);

    Matrix operator T(const Matrix&);
    Matrix operator T(Matrix&);
    double operator T&(int r, int c); //T&
    double operator T(int r, int c); //T

    friend Matrix operator +(const Matrix& a, const Matrix& b);
    friend Matrix operator -(const Matrix& a, const Matrix& b);
    friend Matrix operator *(const Matrix& a, const Matrix& b);

    // add another matrix to this one, changing this
    Matrix operator +=(const Matrix& b);
    // subtract another matrix from this one, changing this
    Matrix operator -=(const Matrix& b);

    vector<double> gaussPartialPivoting(vector<double>& B); // solve (*this)x = B, returning x.

    void gaussPartialPivoting(vector<double>&x, vector<double>& B); // solve (*this)x = B, modifying x that is passed by reference

    vector<double> gaussFullPivoting(vector<double>& B); // solve (*this)x = B, returning x.

    void gaussFullPivoting(vector<double>&x, vector<double>& B); // solve (*this)x = B, modifying x that is passed by reference
    // a to the integer power k
    friend Matrix operator ^(const Matrix& a, int k);
    // write out matrix to a stream
    friend ostream& operator <<(ostream& s, const Matrix& m);
    // read in matrix from a stream
    friend istream& operator >>(istream& s, Matrix& m);

};
#endif //PCA_MATRIX_H
