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

/*
 * @todo: @bug redefinition issue with header when using it with : rows(rows)...
Matrix::Matrix(size_t rows, size_t cols, double val = 0) : rows(rows), cols(cols)
{
    m = new double[rows*cols];
    for (size_t i = 0; i < rows*cols; i++)
        m[i] = val;
}//*/
size_t Matrix::getIdx(size_t r, size_t c)
{
    return ((cols*r) + c);// rows
}
size_t Matrix::getIdx(size_t r, size_t c) const
{
    return ((cols*r) + c);
}

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
/*
 *
 */
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
Matrix& Matrix::operator =(const Matrix& orig)
{
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
Matrix& operator ~(const Matrix& a)
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
            if (b.rows == 1 && sizeb != 1)
            {
                double *arr = new double[a.rows * a.cols];
                for(size_t r = 0; r < a.rows; r++)
                {
                    for(size_t c = 0; c < a.cols; c++)
                        arr[a.getIdx(r, c)] = a(r, c) - b(0, c);//((r*a.rows) + c)
                }
                return Matrix(a.rows, a.cols, arr);
            }
            else if (b.cols == 1 && sizeb != 1)
            {
                double *arr = new double[a.rows * a.cols];
                for(size_t r = 0; r < a.rows; r++)
                {
                    for(size_t c = 0; c < a.cols; c++)
                        arr[a.getIdx(r, c)] = a(r, c) - b(r, 0);//((r*a.rows) + c)
                }
                return Matrix(a.rows, a.cols, arr);
            }
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
    if(a.rows != b.rows || a.cols != b.cols) exit(1);
    double *arr = new double[a.rows * a.cols];
    for(size_t r = 0; r < a.rows; r++)
    {
        for(size_t c = 0; c < a.cols; c++)
            arr[a.getIdx(r, c)] = a(r, c) - b(r, c);//((r*a.rows) + c)
    }
    // no need to delete arr since it becomes nullptr once Matrix is created
    return Matrix(a.rows, b.cols, arr);
}

Matrix operator *(const Matrix& a, const Matrix& b)
{
    if(a.cols != b.rows) exit(1);
    double *arr = new double[a.rows * b.cols];
    for(size_t i = 0; i< (a.rows * b.cols); i++)
    {
        arr[i] = 0;
    }
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


// add another matrix to this one, changing this
Matrix Matrix::operator +=(const Matrix& b)
{
    for(size_t i = 0; i < (this->cols * this->rows); i++)
    {
        this->m[i] += b.m[i];
    }
    return *this;
}
// subtract another matrix from this one, changing this
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



// will return array as [Q, R]
Matrix* Matrix::eigen(const Matrix &A)
{
    const size_t n = A.cols;
    Matrix d, e, Va = A;
    
    for(size_t i = 0; i < A.rows; i++)
    {
        houseHolder(Va, d, e, i);
        ql(Va, d, e, i);
    }
    
    
    Matrix *eig = new Matrix[2];
    eig[0] = Va;
    eig[1] = makeD(d, e);
    
    return eig;
}
// will change Q and R passed in
void Matrix::eigen(const Matrix& A, Matrix& V, Matrix& D)
{
    const size_t n = A.cols;
    Matrix d, e, Va = A;
    for(size_t i = 0; i < n; i++)
    {
        houseHolder(Va, d, e, i);
        ql(Va, d, e, i);
    }
    V = Va;
    D = makeD(d, e);
    
    return;
}

void Matrix::houseHolder(Matrix& A, Matrix& d, Matrix& e, const size_t& row)
{
    // Credit written in GO:https://github.com/skelterjohn/go.matrix/blob/go1/dense_eigen.go
    //  This is derived from the Algol procedures tred2 by
    //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutine in EISPACK.
    
    const size_t n = A.cols;
    for(int i = 0; i < n; i++)
        d(row, i) = A(n-1, i);
    
    // householder reduction to tridiagnal form
    double scale, h, f, g, hh;
    for(size_t i = n-1; i > 0; i--)
    {
        // keep it from overflow
        scale = 0.0, h = 0.0, f = 0.0, g = 0.0;
        for(size_t j = 0; j < i; j++)
            scale += std::abs(d(row, j));
        
        // generate Householder vector
        for(size_t j = 0; j < i; j++)
        {
            d(row, j) /= scale;
            h +=  d(row, j) * d(row, j);
        }
        f = d(row, i-1);
        g = sqrt(h);
        if(f > 0) g = -g;
        e(0, i) = scale * g;
        h = h - f*g;
        d(row, i-1) = f - g;
        for(size_t j = 0; j < i; j++)
            e(0, j) = 0.0;
        // similarity transformation to remaining columns
        for(size_t j =0; j < i; j++)
        {
            f = d(row,j);
            A(j, i) = f;
            g = e(0,j) + A(j, j)*f;
            for(size_t k = 0; k <= i-1; k++)
            {
                g += A(k, j) * d(row,k);
                e(0,k) += A(k, j) * f;
            }
            e(0,j) = g;
        }
        f = 0.0;
        for(size_t j = 0; j < i; j++)
        {
            e(0,j) /= h;
            f += e(0, j) * d(row, j);
        }
        hh = f / (h + h);
        for(size_t j = 0; j < i; j++)
            e(0,j) -= hh * d(row,j);
        for(size_t j = 0; j < i; j++)
        {
            f = d(row, j);
            g = e(0, j);
            for(size_t k = j; k <= i-1; k++)
                A(k,j) -= (f*e(0,k)) + (g*d(row,k));
            d(row,j) = A(i-1, j);
            A(i, j) = 0.0;
        }
        d(row,i) = h;
    }
    
    
    // Accumulate Transformation
    //double h, g;
    for(size_t i = 0; i < n-1; i++)
    {
        A(n-1, i) = A(i, i);
        A(i, i) = 1.0;
        h = d(row, i+1);
        if(h != 0.0)
        {
            for(size_t j = 0; j <= i; j++)
                d(row, j) = A(j, i+1) / h;
            for(size_t j = 0; j <= i; j++)
            {
                g = 0.0;
                for(size_t k = 0; k <= i; k++)
                    g += A(k, i+1) * A(k, j);
                for(size_t k = 0; k <= i; k++)
                    A(k, j) -= g * d(row, k);
            }
        }
        for(size_t j = 0; j <= i; j++)
            A(j, i+1) = 0.0;
        
    }
    for(size_t i = 0; i < n; i++)
    {
        d(row, i) = A(n-1, i);
        A(n-1, i) = 0.0;
    }
    A(n-1, n-1) = 1.0;
    e(0, 0) = 0.0;
}
void Matrix::ql(Matrix& A, Matrix& d, Matrix& e, const size_t& row)
{
    const size_t n = A.cols;
    for(size_t i = 1; i < n; i++)
        e(row,i-1) = e(row, i);
    e(row, n-1);
    size_t m;
    double f, t, tst1 = 0.0, eps = pow(2.0, -52.0);
    for(size_t l = 0; l < n; l++)
    {
        // find small subdianal element;
        t = (std::abs(d(row,l)) + std::abs(e(row,l)));
        tst1 = tst1 > t ? tst1 : t;
        
        for(m = l; m < n; m++)
            if(e(row,m) <= eps * tst1) break;
        // if m == l, d(row,l) is an eigen value
        size_t iter = 0;
        double g, p, r, dl1, h, c ,c2, c3, el1, s, s2;
        if(m > l)
        {
            for(;;)
            {
                iter++;
                // compute implisit shift;
                g = d(row, l);
                p = (d(row, l+1) - g) / (2.0 * e(row, l));
                r = sqrt(p*p + 1.0);
                if(p < 0)
                    r = -r;
                d(row, l) = e(row, l) / (p + r);
                d(row, l+1) = e(row, l) * (p + r);
                dl1 = d(row, l+1);
                h = g - d(row, l);
                for(size_t i = l+2; i < n; i++)
                    d(row, i) -= h;
                f += h;
                
                // implisi QL transformation
                p = (row, m);
                c = c2 = c3 = 1.0;
                el1 = e(row, l+1);
                s = 0.0;
                s2 = 0.0;
                for(size_t i = m - 1; i >= l; i--)
                {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e(row, i);
                    h = c * p;
                    r = sqrt((p * p) + (e(row, i)*e(row, i)));
                    e(row, i+1) = s * r;
                    s = e(row, i)/r;
                    c = p / r;
                    p = c * d(row, i) - (s * g);
                    d(row, i+1) = h + (s*((c*g) + (s*d(row,i))));
                    
                    // accummulate transformation
                    for(size_t j = 0; j < n; j++)
                    {
                        h = A(j, i+1);
                        A(j, i+1) = s*A(j, i) + (c * h);
                        A(j, i) = c * A(j, i) - (s * h);
                    }
                }
                p = -s * s2 * c3 * el1 * e(row, l) / dl1;
                e(row, l) = s*p;
                d(row, l) = c*p;
                
                // check for convergece
                if(!(std::abs(e(row, l) > (eps * tst1))))
                    break;
            }
        }
        d(row, l) = d(row, l) + f;
        e(row, l) = 0.0;
        
    }
    // sort eigen values and corresponding vectors;
    double k, p;
    for(size_t i = 0; i < n-1; i++)
    {
        k = i;
        p = d(row,i);
        for(size_t j = i +1; j < n; j++)
        {
            if(d(row, j) < p)
            {
                k = j; p = d(row, j);
            }
        }
        if(k != i)
        {
            d(row, k) = d(row, i);
            d(row, i) = p;
            for(size_t j = 0; j < n; j++)
            {
                p = A(j, i);
                A(j, i) = A(j, k);
                A(j, k) = p;
            }
        }
    }
}
Matrix Matrix::makeD(const Matrix& d, const Matrix& e)
{
    const size_t n = d.cols;
    int sign, offset;
    Matrix X(n, n);// gives me of zeros
    for(size_t i = 0; i < n; i++)
    {
        sign = e(i, i);
        X(i, i) = d(i, i); // assuming its just one row according to caller eigen()
        offset = sign < 0? -1: 1;
        X(i, i + offset) = e(i, 1);
    }
    return X;
}
/*
// will return array as [Q, R]
Matrix* Matrix::qrDecomp(const Matrix &A)
{
    return new Matrix();
}
// will change Q and R passed in
void Matrix::qrDecomp(const Matrix& A, Matrix& Q, Matrix& R)
{
    const size_t m = A.rows;
    int sign = A(0,0) >= 0 ? 1:-1;
    
}

// will return array as [Q, R]
Matrix Matrix::houseHolder(const Matrix& A)
{
    return Matrix();
}
*/

//read CSV file into matrix object
void Matrix::readFile(std::string filename)
{
    std::vector<double> input;
	std::fstream inFile;
	std::string line;

	inFile.open(filename);
    try {
        if(!inFile.is_open()) throw std::string("file did not open\n");
        
    }
    catch(std::string msg)
    {
        std::cout << msg;
    }
    size_t r = 0;
    size_t c = 0;

	//get each line of the file, then push each value from every line onto
	//the back of a vector
	while (!inFile.eof() && getline(inFile, line, '\n'))
	{
		std::stringstream ss(line);
		c=1;
		while(ss)
		{
			std::string s;
			if (!getline(ss, s, ','))
				break;
			double d = std::stod(s);
			input.push_back(d);
			c++;
		}
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

// a to the integer power k
/*friend Matrix& Matrix::operator ^(const Matrix& a, size_t k)
{
    // @todo: implement function
}*/

// write out matrix to a stream
std::ostream& operator<<(std::ostream& s, const Matrix& m)
{
    double val;
    for(size_t i = 0; i < m.rows; i++)
    {
        for(size_t j = 0; j < m.cols; j++)
        {
            val = m(i, j);
            s << val << ',';
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
