/** @file Matrix.cpp
 *  @brief Principal Component Analysis Class.
 *
 *  Contains class to calculate most significant feature for a given class
 *  and inputs for those features.
 *
 *  @author Ariel Feliz(aafeliz)
 *  @author Elvin Abreu(elvinabreu)
 *  @author Dov Kruger
 *  @date 11/27/16
 *  @bug No known bugs.
 *  @todo implement empty methods
 */
#include "../include/Matrix.h"
#include <cstdint>
#include <iostream>
#include <fstream>
#include <cassert>

/*
 * @todo: @bug redefinition issue with header when using it with : rows(rows)...
Matrix::Matrix(size_t rows, size_t cols, double val = 0) : rows(rows), cols(cols)
{
    m = new double[rows*cols];
    for (size_t i = 0; i < rows*cols; i++)
        m[i] = val;
}//*/

/* gets the an (i,j) index from the matrix	*/
size_t Matrix::getIdx(size_t r, size_t c)
{
    return ((cols*r) + c);// rows
}
size_t Matrix::getIdx(size_t r, size_t c) const
{
    return ((cols*r) + c);
}
size_t Mat::getIdx(const size_t& Rows, const size_t& Cols, const size_t& r, const size_t& c)
{
    return (Cols * r) + c;
}

/**@bug: delete cannot be used when arr-> is pointing at an array that did not use new*/
Matrix::Matrix(size_t r, size_t c, double *arr)
{
    rows = r, cols = c;
    m = new double[rows * cols];
    for(size_t i = 0; i < (rows * cols); i++)
    {
        m[i] = arr[i];
    }
    delete[] arr;
}
Matrix::Matrix(size_t rows, size_t cols, double **data)
{
    this->rows = rows;
    this->cols = cols;
    m = new double[rows * cols];
    for(size_t r = 0; r < rows; r++)
    {
        for(size_t c = 0; c < cols; c++)
            m[getIdx(r, c)] = data[r][c];
    }
}

Matrix::~Matrix()
{
    delete[] m;
}

/* make a copy of given matrix	*/
Matrix::Matrix(const Matrix& orig) : rows(orig.rows), cols(orig.cols)
{
    /**@todo: should it delete m first, possible memory leak
     */
    m = new double[rows*cols];
    for (size_t i = 0; i < rows*cols; ++i)
        m[i] = orig.m[i];
}


/*Matrix::Matrix(Matrix&& orig) : rows(orig.rows), cols(orig.cols), m(orig.m)
{
    // might want to keep original
    //orig.m = nullptr;
}*/

/*	assign a matrix to be equal to another	*/
Matrix& Matrix::operator =(const Matrix& orig)
{
	//	if current matrix is not equal to the one being passed in,
	//	copy all the contents of the given matrix into the current one
    if (this != &orig) {
        delete[] this->m;
        this->m = new double[orig.cols * orig.rows];
        for(size_t i = 0; i < (orig.cols * orig.rows); i++)
        {
            this->m[i] = orig.m[i];
        }
        this->cols = orig.cols;
        this->rows = orig.rows;
    }
    return *this;
}

/*	operator function to easily access a certain index of the matrix	*/
double& Matrix::operator ()(size_t r, size_t c)
{
    return m[getIdx(r, c)];
}
double& Matrix::operator()(size_t r, size_t c) const
{
    return m[getIdx(r, c)];
}

/*
double& Matrix::operator[](size_t r, size_t c)
{
    return (this->m[getIdx(r, c)]);
}*/
inline void Matrix::setValue(size_t r, size_t c, double val)
{
    this->m[getIdx(r, c)] = val;
}
/*
Matrix Matrix:: operator T&(size_t r, size_t c)
{
    // @todo: implement function
    Matrix mx;
    return mx;
}
Matrix Matrix:: operator T(size_t r, size_t c) const
{
    // @todo: implement function
    Matrix mx;
    return mx;
}
double Matrix:: operator T&(size_t r, size_t c)
{
    return m[r*cols + c];
}
double Matrix:: operator T(size_t r, size_t c) const
{
    return m[r*cols + c];
}
*/

/*	transpose a given matrix	*/
Matrix& operator ~(const Matrix& a)
{
    const size_t size = a.rows * a.cols;
    double *arr = new double[size];
    
    size_t r = 0, c = 0, tidx = 0;
    
    /* go thru whole array and find the
    correct index in our "matrix" (actually
    an array) to place the value */
    for(size_t i = 0; i < size; i++)
    {
        r = i / a.cols;
        c = i % a.cols;
        tidx = (c*a.rows) + r;
        arr[tidx] = a.m[i];
    }
    
    //set our matrix to the built array
    return *new Matrix(a.cols, a.rows, arr);
}
/*
Matrix operator ~(Matrix a)
{
    const size_t size = a.rows * a.cols;
    double *arr = new double[size];

    size_t r = 0, c = 0, tidx = 0;
    for(size_t i = 0; i < size; i++)
    {
        r = i / a.cols;
        c = i % a.cols;
        tidx = (c*a.rows) + r;
        arr[tidx] = a.m[i];
    }

    return Matrix(a.cols, a.rows, arr);
}*/

void operator ~(Matrix& a)
{
    const size_t size = a.rows * a.cols;
    double *arr = new double[size];
    
    for(size_t i = 0; i < size; i++)
        arr[i] = a.m[i];

    size_t r = 0, c = 0, tidx = 0;
    for(size_t i = 0; i < size; i++)
    {
        r = i / a.cols;
        c = i % a.cols;
        tidx = (c*a.rows) + r;
        a.m[tidx] = arr[i];
    }

    //swap rows with columns
    size_t temp = a.cols;
    a.cols = a.rows;
    a.rows = temp;
    
    delete[] arr;
    return;
    //return Matrix(a.cols, a.rows, arr);
}
/*double Matrix::operator ~(size_t r, size_t c)
{
    //return this->m[r*cols + c];
    return 0.0;
}*/

/*	add two matrices together	*/
Matrix operator +(const Matrix& a, const Matrix& b)
{
    if(a.rows != b.rows || a.cols != b.cols) exit(1);
    double *arr = new double[a.rows * a.cols];
    for(size_t r = 0; r < a.rows; r++)
    {
        for(size_t c = 0; c < a.cols; c++)
        {
            arr[a.getIdx(r, c)] = a(r, c) + b(r, c);
        }
    }
    // no need to delete arr since it becomes nullptr once Matrix is created
    return Matrix(a.rows, b.cols, arr);
}

/*	subtract one matrix from another	*/
Matrix operator -(const Matrix& a, const Matrix& b)
{
    size_t sizea = a.rows*a.cols;
    size_t sizeb = b.rows*b.cols;
    
    //can't subtract large b from small a
    if (sizeb > sizea) exit(1);
    
    else if (sizea != sizeb)
    {
        if (b.rows == a.rows || b.cols == a.cols)
        {
        	//if b has one row but not just one element
            if (b.rows == 1 && sizeb != 1)
            {
                double *arr = new double[a.rows * a.cols];
                for(size_t r = 0; r < a.rows; r++)
                {
                	//subtract each of the elements in the 0th row
                    for(size_t c = 0; c < a.cols; c++)
                        arr[a.getIdx(r, c)] = a(r, c) - b(0, c);//((r*a.rows) + c)
                }
                return Matrix(a.rows, a.cols, arr);
            }
            //if b has one column but not just one element
            else if (b.cols == 1 && sizeb != 1)
            {
                double *arr = new double[a.rows * a.cols];
                for(size_t r = 0; r < a.rows; r++)
                {
                	//subtract each of elements in the 0th column
                    for(size_t c = 0; c < a.cols; c++)
                        arr[a.getIdx(r, c)] = a(r, c) - b(r, 0);//((r*a.rows) + c)
                }
                return Matrix(a.rows, a.cols, arr);
            }
            //if b is just one element
            else if (sizeb == 1)
            {
                if (a.cols == 1)
                {
                    double *arr = new double[a.rows * a.cols];
                    for(size_t r = 0; r < a.rows; r++)
                    {
                        for(size_t c = 0; c < a.cols; c++)
                            arr[a.getIdx(r, c)] = a(r, 0) - b(0, 0);//((r*a.rows) + c)
                    }
                    return Matrix(a.rows, a.cols, arr);
                }
                if (a.rows == 1)
                {
                    double *arr = new double[a.rows * a.cols];
                    for(size_t r = 0; r < a.rows; r++)
                    {
                        for(size_t c = 0; c < a.cols; c++)
                            arr[a.getIdx(r, c)] = a(0, c) - b(0, 0);//((r*a.rows) + c)
                    }
                    return Matrix(a.rows, a.cols, arr);
                }
            }
        }
        else { /**@todo:subtracting smaller matrix and b not a vector*/ }
    }

    if(a.rows != b.rows || a.cols != b.cols)
    	exit(1);
    double *arr = new double[a.rows * a.cols];
    for(size_t r = 0; r < a.rows; r++)
    {
        for(size_t c = 0; c < a.cols; c++)
            arr[a.getIdx(r, c)] = a(r, c) - b(r, c);//((r*a.rows) + c)
    }
    // no need to delete arr since it becomes nullptr once Matrix is created
    return Matrix(a.rows, b.cols, arr);
}

/*	multiply two matrices together	*/
Matrix operator *(const Matrix& a, const Matrix& b)
{
    //if this isn't true, matrix multiplication not valid
    if(a.cols != b.rows)
    	exit(1);
    //create array of doubles and initialize to 0
    double *arr = new double[a.rows * b.cols];
    for(size_t i = 0; i< (a.rows * b.cols); i++)
    {
        arr[i] = 0;
    }
    /*each i,j element is found by multiplying the entries
      across row i of A by the entries down column j of B and
      summing them from k=1 to m (m is a.cols or b.rows) */
    for(size_t ra = 0; ra < a.rows; ra++)
    {
        for(size_t cb = 0; cb < b.cols; cb++)
        {
            for(size_t ca = 0; ca < a.cols; ca++)
                arr[((ra * a.rows) + cb)] += a(ra, ca) * b(ca, cb);
        }
    }
    // no need to delete arr since it becomes nullptr once Matrix is created
    return Matrix(a.rows, b.cols, arr);
}
/**@brief: scalar to matrix multiplication
 
 */

/*	multiply a matrix by a scalar	*/
Matrix operator *(const double& a, const Matrix& b)
{
    double *arr = new double[b.rows * b.cols];
    for(size_t i = 0; i< (b.rows * b.cols); i++)
    {
        arr[i] = 0;
    }
    for(size_t r = 0; r < b.rows; r++)
    {
        for(size_t c = 0; c < b.cols; c++)
        {
        	//multiply each element by scalar
            arr[b.getIdx(r, c)] = a * b(r, c);
        }
    }
    // no need to delete arr since it becomes nullptr once Matrix is created
    return Matrix(b.rows, b.cols, arr);
}


/*	add another matrix to this one, changing this	*/
Matrix Matrix::operator +=(const Matrix& b)
{
    for(size_t i = 0; i < (this->cols * this->rows); i++)
    {
        this->m[i] += b.m[i];
    }
    return *this;
}

/*	add constant to every element in this matrix, changing this	*/
Matrix Matrix::operator +=(const double& b)
{
    for(size_t i = 0; i < (this->cols * this->rows); i++)
    {
        this->m[i] += b;
    }
    return *this;
}

/*	subtract another matrix from this one, changing this	*/
Matrix Matrix::operator -=(const Matrix& b)
{
    for(size_t i = 0; i < (this->cols * this->rows); i++)
    {
        this->m[i] -= b.m[i];
    }
    return *this;
}
/////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

/*
 *  IMPORTANT CHECKS:
     - maxElement function
        + the for loop size might just be n
     - rotate
        + before the cases check order of ops in A(R, R) being set
 
 */
/**@brief: Jacobian eigen decomposition helper function find max value
        * Will retrieve value and location of largest value of top half of matrix
        * Returns double array with [maxVal, Row, Col]
 */
double* Mat::thMaxElement(const Matrix& A)
{
    double *all = new double[3];
    const size_t n = A.cols;
    for(size_t r = 0; r < n-1; r++)
    {
        for(size_t c = r+1; c < n; c++)
        {
            if(std::abs(A(r, c)) >= all[0])
            {
                all[0] = std::abs(A(r,c));
                all[1] = r;
                all[2] = c;
            }
        }
    }
    return all;
}

void Mat::mRotate(Matrix& A, Matrix& p, const u_int32_t& R, const u_int32_t& C)
{
    /** goal is to rotate to make A(R, C) = 0 then update the transformation matrix P
        The rotation distributes the coeficient to make it possible to have the off-diagnal
        values queal 0, while making diagnal gets bigger.
        The algorithm relies on having R and C being the location in A where the Max value is located outside the diagnol
        It is also the focus in which the more iterations the function is used the more accurate the
        transformation matrix becomes.

    */
    const size_t n = A.cols;
    double aDiff, t, phi, c, s, tau, temp;
    // getting the differences
    aDiff = A(C, C) - A(R, R);

    // A(R, C) is where the max was found off the diagonal
    if(std::abs(A(R, C)) < std::abs(aDiff))
        t = A(R, C)/aDiff;
    else
    {
        phi = aDiff / (2.0* A(R, C));
        t = 1.0 / (std::abs(phi) + sqrt((phi*phi) +1));
        if(phi < 0.0)
            t = -t;
    }

    c = 1.0 / sqrt((t * t) + 1.0);
    s = t * c;
    tau = s / (1.0 + c);
    temp = A(R, C);
    A(R, C) = 0.0;
    A(R, R) = A(R, R) - (t * temp);
    A(C, C) = A(C, C) + (t * temp);

    /*
     *  Cases distribute the new values using the coeficients calculated above
     */
    /** changing all as shown
     *        C/R   R/C
     *        |     |
     *        |     |
     *        |     |
     *        |     |
     *        |     |
     *
     Case of i < R */
    for(size_t i = 0; i < R; i++)
    {
        temp = A(i, R);
        A(i, R) = temp - (s * (A(i, C) + (tau * temp)));
        A(i, C) = A(i, C) + (s * (temp - (tau * A(i, C))));
    }

    /**changing the values two lines; rotation
    *                R-----
    *                      |
    *                      |
    *                      |
     Case of R < i < C  */
    for(size_t i = R + 1; i < C; i++)
    {
        temp = A(R, i);
        A(R, i) = temp - (s * (A(i, C) + (tau * A(i, C))));
        A(i, C) = A(i, C) + (s * (temp - (tau * A(i, C))));
    }
    /** changing two rows as shown
     *
     *    R/C--------------
     *
     *    C/R--------------
     *
     Case of i > C */
    for(size_t i = C + 1; i < n; i++)
    {
        temp = A(R, i);
        A(R, i) = temp - (s * (A(C, i) + (tau * temp)));
        A(C, i) = A(C, i) + (s * (temp - (tau * A(C, i))));
    }
    /// Update trans matrix
    for(size_t i = 0; i < n; i++)
    {
        temp = p(i, R);
        p(i, R) = temp - (s * (p(i, C) + (tau * p(i, R))));
        p(i, C) = p(i, C) + (s * (temp - (tau * p(i, C))));
    }
}

Matrix* Matrix::jacobian_eig()
{
    try
    {
        Matrix cpA(rows, cols, m);
        const size_t n = cols;
        // set number of rotations
        const size_t maxNrot = 5 * n * n;
        // init the transformation matrix
        Matrix p(n, n);
        for(size_t i = 0, j = 0; i < n; i++, j++)
            p(i,j) = 1;
        // jacobi rotation loop
        double *aMax;
        for(size_t i = 0; i < maxNrot; i++)
        {
            aMax = Mat::thMaxElement(cpA);
            if(aMax[0] < 1.0e-10)
            {
                // diagnal from A that is to be eigen values
                Matrix aVals(1,n);
                for(size_t j = 0; j < n; j++)
                    aVals(0,j) = cpA(j, j);
                // vector will be p
                Matrix aVect(p);
                Matrix* all = new Matrix[2];
                all[0] = aVals;
                all[1] = aVect;
                return all;
            }
            Mat::mRotate(cpA, p, u_int32_t(aMax[1]), u_int32_t(aMax[2]));
        }
        std::string e = "no converge";
        throw(e);
    }
    catch(std::string error)
    {
        std::cout << error << '\n';
        return nullptr;
    }
}

Matrix* Mat::jacobian_eig(const Matrix& A)
{
    try
    {
        Matrix cpA(A);
        const size_t n = A.cols;
        // set number of rotations
        const size_t maxNrot = 5 * n * n;
        // init the transformation matrix
        Matrix p(n, n);
        for(size_t i = 0, j = 0; i < n; i++, j++)
            p(i,j) = 1;
        // jacobi rotation loop
        double *aMax;
        for(size_t i = 0; i < maxNrot; i++)
        {
            aMax = Mat::thMaxElement(cpA);
            if(aMax[0] < 1.0e-10)
            {
                // diagnal from A that is to be eigen values
                Matrix aVals(1,n);
                for(size_t j = 0; j < n; j++)
                    aVals(0,j) = cpA(j, j);
                // vector will be p
                Matrix aVect(p);
                Matrix* all = new Matrix[2];
                all[0] = aVals;
                all[1] = aVect;
                return all;
            }
            Mat::mRotate(cpA, p, u_int32_t(aMax[1]), u_int32_t(aMax[2]));
        }
        std::string e = "no converge";
        throw(e);
    }
    catch(std::string error)
    {
        std::cout << error << '\n';
        return nullptr;
    }
}


void Matrix::sortRow(size_t r)
{
    Matrix t = *this;
    size_t i, j, key;
    for(j = 1; j < cols; j++)    // Start with 1 (not 0)
    {
        key = t(r,j);
        for(i = j - 1; (i >= 0) && (t(r,i) < key); i--)   // Smaller values move up
        {
            t(r,i+1) = t(r,i);
        }
        t(r,i+1) = key;    //Put key into its proper location
    }
    *this = t;
    return;
}
void Mat::sortRowAB_BasedOnA(const size_t& r, Matrix& A, Matrix& B)
{
    size_t i, j, key;
    double arr[B.rows];
    for(j = 1; j < A.cols; j++)    // Start with 1 (not 0)
    {
        key = A(r,j);
        for(size_t k = 0; k < B.rows; k++)
            arr[k] = B(k, j);
        for(i = j - 1; (i >= 0) && (A(r,i) < key); i--)   // Smaller values move up
        {
            A(r,i+1) = A(r,i);
            for(size_t k = 0; k < B.rows; k++)
                B(k,i+1) = B(k, i);
        }
        A(r,i+1) = key;    //Put key into its proper location
        for(size_t k = 0; k < B.rows; k++)
            B(k,i+1) = arr[k];
    }
    return;
}
size_t Mat::getMaxIdx(const Matrix &A)
{
    size_t maxIdx = 0;//A(0,0);
    double maxVal = A(0,0);
    for(size_t i = 0; i < A.cols; i++)
    {
        if(A(0, i) > maxVal)
        {
            maxVal = A(0, i);
            maxIdx = i;
        }
        
    }
    return maxIdx;
}


//read CSV file into matrix object
void Matrix::readFile(std::string filename)
{
    std::vector<double> input;
	std::fstream inFile;
	std::string line;
	inFile.open(filename);
    try
    {
        if(!inFile.is_open())
        	throw std::string("File did not open successfully.\n");
    }
    catch(std::string msg)
    {
        std::cout << msg;
    }

    //going to use r and c to determine the # of rows and cols
    size_t r = 0;
    size_t c = 0;

	//get each line of the file, then push each value from
	//every line onto the back of a vector
	while (!inFile.eof() && getline(inFile, line, '\n'))
	{
		std::stringstream ss(line);
		c=1;
		while(ss)
		{
			std::string s;
			//delimit on ',' (since it's a csv)
			if (!getline(ss, s, ','))
				break;
			//convert current string number val to double
			//and push it back onto the vector
			double d = std::stod(s);
			input.push_back(d);

			//increment number of columns the read matrix has
			c++;
		}
		//after every line is complete, increment # of rows
		r++;
	}
	inFile.close();
	rows = r;
	cols = c-1;	//for some reason it ends up being one more than it should

	//copy input vector to m so matrix class can work
	m = new double[rows*cols];
    for (size_t i = 0; i < rows*cols; ++i)
        m[i] = input[i];
}

//generate a CSV file from a matrix
void Matrix::writeFile(std::string filename)
{
	std::ofstream outFile;
	outFile.open(filename);
    double val;

    /*	for each val in our matrix, output the number
    	followed by a comma, except the last column
    	whose values should be follow with a \n instead
    	of a comma	*/
    for(size_t i = 0; i < this->rows; i++)
    {
        for(size_t j = 0; j < this->cols-1; j++)
        {
            val = m[getIdx(i,j)];
            outFile << val << ',';

        }
        val = m[getIdx(i,cols-1)];
        outFile << val << '\n';
    }
    outFile.close();
}

Matrix Matrix::getColumn(size_t col)
{
    double *arr = new double[rows];
    for(size_t i = 0; i < rows; i++)
    {
        arr[i] = this->m[getIdx(i, col)];
    }
    return Matrix(rows, 1, arr);
}
Matrix Matrix::getRow(size_t row)
{
    double *arr = new double[cols];
    for(size_t i = 0; i < cols; i++)
    {
        arr[i] = this->m[getIdx(row, i)];
    }
    return Matrix(1, cols, arr);
}

/*	add a row to the given matrix	*/
void Matrix::appendRow(const Matrix& B)
{
    const size_t oldSize = rows * B.cols;
    const size_t newSize = B.cols + oldSize;
    
    double *temp = new double[newSize];
    for(size_t n = 0; n < newSize; n++)
    {
        if(n < oldSize)
            temp[n] = m[n];
        else
            temp[n] = B(0,n-oldSize);
    }
    delete[] m;
    m = temp;
    rows++;
    cols = B.cols;
    
    
}
/*	add a column to the given matrix	*/
void Matrix::appendCol(const Matrix& B)
{
    const size_t oldCols = cols;
    cols++;
    rows = B.rows;
    const size_t newSize = cols * B.rows;
    double *temp = new double[newSize];
    for(size_t r = 0; r < B.rows; r++)
    {
        for(size_t c = 0; c < cols; c++)
        {
            if(c < oldCols)
                temp[getIdx(r, c)] = m[Mat::getIdx(rows, oldCols, r, c)];
            else
                temp[getIdx(r, c)] = B(r,0);
        }
    }
    delete[] m;
    m = temp;
    
    
}

// a to the integer power k
/*friend Matrix& Matrix::operator ^(const Matrix& a, size_t k)
{
    // @todo: implement function
}*/

// write out matrix to a stream
std::ostream& operator<<(std::ostream& s, const Matrix& m)
{
    double val;
    s.precision(4);
    for(size_t i = 0; i < m.rows; i++)
    {
        for(size_t j = 0; j < m.cols; j++)
        {
            val = m(i, j);
            s << std::setw(12) << val << ',';
        }
        s << '\n';
    }
    return s;
}

/* read in matrix from a stream
std::istream& operator >>(std::istream& s, Matrix& m)
{
    // @todo: implement function

}*/

inline void Matrix::setRows(size_t r)
{
    rows = r;
}
inline void Matrix::setCols(size_t c)
{
    cols = c;
}
