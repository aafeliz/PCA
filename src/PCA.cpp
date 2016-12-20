/** @file PCA.cpp
 *  @brief Principal Component Analysis Class.
 *
 *  Contains class to calculate most significant feature for a given class
 *  and inputs for those features.
 *
 *  @author Ariel Feliz(aafeliz)
 *  @author Elvin Abreu(elvinabreu)
 *  @date 11/26/16
 *  @bug No known bugs.
 *  @todo Need to create methods for eigen decomposition. Providing PCA class with eigen value & vector.
 */
#include "../include/PCA.h"
//#include <cstdarg.h> //va_list, va_start, va_end
#include <assert.h>
//FeatureData::FeatureData(std::string name, Matrix data): FeatureName(name), data(data) {}


PCA::PCA()
{
	numFeatures = 0;
}



/**@brief: constructor for PCA when features and data are being passed in
 */
PCA::PCA(const Matrix& featuresData)
{
	numFeatures = featuresData.rows;
    this->featuresData = featuresData;
}




// will return array as [Q, R]
Matrix* PCA::eigenHH()
{
    A = sMat;
    const size_t n = A.cols;
    
    // d and e are length of cols,
    Matrix d(n, n, double(0)), e(n, n, double(0)), Va = A;
    Matrix *eig = new Matrix[2];
    eig[1] = Matrix(n, n, double(0));
    for(size_t i = 0; i < A.rows-1; i++)
    {
        houseHolder(Va, d, e, i);
        ql(Va, d, e, i);
        eig[1] += makeD(d, e, i);
    }
    std::cout << std::endl;
    std::cout << e;
    eig[0] = Va;
    
    
    return eig;
}
// will change Q and R passed in
void PCA::eigenHH(Matrix& V, Matrix& D)
{
    A = sMat;
    const size_t n = A.cols;
    Matrix d, e, Va = A;
    for(size_t i = 0; i < n; i++)
    {
        houseHolder(Va, d, e, i);
        ql(Va, d, e, i);
        D = makeD(d, e, i);
    }
    V = Va;
    
    return;
}

void PCA::eigenJacobian()
{
    Matrix *eigens = Mat::jacobian_eig(covMat);
    eigenVals = eigens[0];
    eigenVect = eigens[1];
}
void PCA::calcALL()
{
    calcStats();
    calcEigen();
    calcPCA();
}
void PCA::calcPCA()
{
    
    //std::cout << "Xi_mu\n" << xi_mu << '\n';
    const size_t MAX = Mat::getMaxIdx(eigenVals);
    const Matrix maxVec = eigenVect;//.getColumn(MAX);
    //std::cout << "maxVec\n" << maxVec << '\n';
    ai = ~maxVec * xi_mu;
    ai = ai.getRow(0);
    const Matrix teVect = eigenVect.getColumn(MAX);
    //std::cout << "ai\n" << ai << '\n';
    double aii;
    Matrix x_bari;



    /**/
    Mat::sortRowAB_BasedOnA(0, eigenVals, eigenVect);
    for(size_t c = 0; c < ai.cols; c++)
    {
        aii = ai(0,c);
        //std::cout << "aiii :\n" << aii << '\n';
        x_bari = mu + (aii * eigenVect.getColumn(c));//mu + (aii *
        x_bar.appendCol(x_bari);
        //std::cout << "x_bar :\n" << x_bar << '\n';
    }
    
    /**/



    //const Matrix e = eigenVect;
    //std::cout << e << "\n";
    //x_bar = ~e * featuresData;
    //std::cout << x_bar << '\n';
    
    
    /*
        
        ai = ~teVect * xi_mu;
        //1.060	-3.889 5.303	-1.060	-2.474	3.889	-4.596	1.767
        std::cout << "ai: \n" << ai << '\n';
        Matrix aiVect = (teVect * ai);
        aiVect += mu;
        std::cout << "aiVect\n" << aiVect << '\n';
        x_bar = mu + (ai * teVect);
        
        
        //1.7500   -1.7500    4.7500    0.2500   -0.7500    3.7500   -2.2500    2.2500
        //1.2500   -2.2500    4.2500   -0.2500   -1.2500    3.2500   -2.7500    1.7500
        std::cout << "xi_bar: \n" << x_bar << '\n';
         */
    
}

void PCA::calcEigen()
{
    eigenJacobian();
}
void PCA::outputALL()
{
    outputData();
    outputStats();
    outputEigen();
    outputPCA();
    x_bar.writeFile("output.csv");
    eigenVals.writeFile("eigenvals.csv");
    eigenVect.writeFile("eigenvects.csv");
}
void PCA::outputEigen()
{
    outputEigVals();
    std::cout << '\n';
    outputEigVect();
}

void PCA::outputEigVect()
{
    std::cout << "Eigen Vectors:\n" << eigenVect;
    
}

void PCA::outputEigVals()
{
    std::cout << "Eigen Values:\n" << eigenVals;
}
void PCA::outputAi()
{
    std::cout << "Ai's Values:\n" << ai;
}
void PCA::outputX_bar()
{
    std::cout << "Xi_bar Values:\n" << x_bar;
}
void PCA::outputPCA()
{
    outputAi();
    std::cout << '\n';
    outputX_bar();
}
/*
 * NOT WORKING !!!!!
 */
void PCA::houseHolder(Matrix& A, Matrix& d, Matrix& e, const size_t& row)
{
    // Credit written in GO:https://github.com/skelterjohn/go.matrix/blob/go1/dense_eigen.go
    //  This is derived from the Algol procedures tred2 by
    //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutine in EISPACK.
    
    const size_t n = A.cols;
    for(size_t i = 0; i < n; i++)
        d(row, i) = A(n-1, i);
    
    // householder reduction to tridiagnal form
    double scale, h, f, g, hh;
    for(size_t i = n-1; i > 0; i--)
    {
        // keep it from overflow
        scale = 0.0, h = 0.0, f = 0.0, g = 0.0;
        for(size_t j = 0; j < i; j++)
            scale += std::abs(d(row, j));
        
        if(scale == 0.0)
        {
            e(row, i) = d(row, i-1);
            for(size_t j = 0; j < i; j++)
            {
                d(row, j) = A(i-1, j);
                A(i, j) = 0.0;
                A(j, i) = 0.0;
            }
        }
        else
        {
            // generate Householder vector
            for(size_t j = 0; j < i; j++)
            {
                d(row, j) /= scale;
                h +=  d(row, j) * d(row, j);
            }
            f = d(row, i-1);
            g = sqrt(h);
            if(f > 0) g = -g;
            e(row, i) = scale * g;
            h = h - f*g;
            d(row, i-1) = f - g;
            for(size_t j = 0; j < i; j++)
                e(row, j) = 0.0; // all e's might be at e(j, j) not e(row, j)
            
            // similarity transformation to remaining columns
            for(size_t j =0; j < i; j++)
            {
                f = d(row,j);
                A(j, i) = f;
                g = e(row,j) + A(j, j)*f;
                for(size_t k = 0; k <= i-1; k++)
                {
                    g += A(k, j) * d(row,k);
                    e(row,k) += A(k, j) * f;
                }
                e(row,j) = g;
            }
            f = 0.0;
            for(size_t j = 0; j < i; j++)
            {
                e(row,j) /= h;
                f += e(row, j) * d(row, j);
            }
            hh = f / (h + h);
            for(size_t j = 0; j < i; j++)
                e(row,j) -= hh * d(row,j);
            for(size_t j = 0; j < i; j++)
            {
                f = d(row, j);
                g = e(row, j);
                for(size_t k = j; k <= i-1; k++)
                    A(k,j) -= (f*e(row,k)) + (g*d(row,k));
                d(row,j) = A(i-1, j);
                A(i, j) = 0.0;
            }
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
    e(row, 0) = 0.0;
}
void PCA::ql(Matrix& A, Matrix& d, Matrix& e, const size_t& row)
{
    const size_t n = A.cols;
    for(size_t i = 1; i < n; i++)
        e(row,i-1) = e(row, i);
    e(row, n-1) = 0.0;
    size_t m;
    double f, t, de, tst1 = 0.0, eps = pow(2.0, -52.0);
    for(size_t l = 0; l < n; l++)
    {
        // find small subdianal element;
        de = std::abs(d(row,l));
        de += std::abs(e(row,l));
        t = de;
        tst1 = tst1 > t ? tst1 : t;
        m = l;
        while(m < n)
        {
            if(e(row,m) <= eps * tst1) break;
            m++;
        }
        // if m == l, d(row,l) is an eigen value
        size_t iter = 0;
        double g, p, r, dl1, h, c ,c2, c3, el1, s, s2, conL, conR;
        if(m > l)
        {
            iter = 0;
            for(;;)
            {
                iter = iter +1;
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
                p = d(row, m);
                c = 1.0;
                c2 = c;
                c3 = c;
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
                    p = (c * d(row, i)) - (s * g);
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
                conL = std::abs(e(row, l));
                conR =eps * tst1;
                if(!(conL > conR))
                    break;
            }
        }
        d(row, l) = d(row, l) + f;
        e(row, l) = 0.0;
        
    }
    // sort eigen values and corresponding vectors;
    /*
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
     */
}
/**@todo: needs to be fixed
 */
Matrix PCA::makeD(const Matrix& d, const Matrix& e, const size_t& r)
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

/*void PCA::passFeaturesData(int numFeatures, ...)
{
    va_list args;
    va_start(args, numFeatures);
    this->numFeatures += numFeatures;
    //assert(typeid(a) == typeid(double));
    for (int i = 0; i < numfeatures; ++i)
    {
        FeatureData feature = va_arg(args, FeatureData);
        this->featuresData.push_back(feature);
    }

    va_end(ap);
}*/
//@todo need to fix feature data class
/*void PCA::passFeaturesData(vector<FeatureData> features)
{
    if(this->inputData.empty())
    {
        this->featuresData = features;
        this->numFeatures = features.size();
    }
    else
    {
        auto it = this->inputdata.end();
        this->featuresData.insert(it, features.begin(), features.endl());
    }
}*/
void PCA::passFeaturesData(Matrix features)
{
    this->featuresData = features;
}

void PCA::calcMeans()///O(numFeatures*numInputsPerFeature)
{

    /*for(auto it : this->featuresData) //for(auto it = this->inputData.begin(); it != this->inputData.end(); ++it)
    {
        double sum = 0;
        for(it2 : it.data)
        {
            sum += it2;
        }
        double avg = sum/it.data.size();
        this->mu.push_back(avg);
    }*/
    double *arr = new double[numFeatures];
    for(size_t r = 0; r < featuresData.rows; r++)
    {
        double featSum = 0;
        for(size_t c = 0; c < featuresData.cols; c++)
            featSum += featuresData(r, c);
        double avg = featSum/((double)featuresData.cols);
        arr[r] = avg;

    }
    //Matrix m(numFeatures, 1, arr);
    mu = Matrix(numFeatures, 1, arr);
}


/**@field: calculates the scattering matrix then converts to covarience by div by N-1
  *@bug: friend operator wont take matrix as const matrix& unless it is const
  *
 */

void PCA::calcScatterMatrix()
{
    
    Matrix scatterMat(mu.rows, mu.rows, 0.0);
    xi_mu = Matrix(0, 0, 0.0);
    for(size_t c = 0; c < featuresData.cols; c++)
    {
        // calc all xi - mu
        const Matrix xi = featuresData.getColumn(c);
        //xi = featuresData.getColumn(c);
        //xi_mu = xi - mu;
        const Matrix txi_mu = xi - mu;
        xi_mu.appendCol(txi_mu);
        //std::cout << "txi_mu " << c << ":\n" << txi_mu;
        //std::cout << "xi_mu " << c << ":\n" << xi_mu;
        // calc sum((xi-mu) * (xi-mu))
        Matrix xi_mu2 = txi_mu * ~(txi_mu);
        scatterMat += xi_mu2;
    }

    /*
    Matrix scatterMat(mu.rows, mu.rows, 0.0);
    //Matrix xi;
    xi_mu = Matrix(0,0,0.0);
    
    for(size_t r = 0; r < featuresData.rows; r++)
    {
        // calc all xi - mu
        const Matrix xi = featuresData.getRow(r);
        //xi = featuresData.getColumn(c);
        //xi_mu = xi - mu;
        const Matrix txi_mu = xi - mu;
        xi_mu.appendRow(txi_mu);
        std::cout << "xi_mu " << r << ":\n" << xi_mu;
        
        // calc sum((xi-mu) * (xi-mu))
        Matrix xi_mu2 = txi_mu * ~(txi_mu);
        scatterMat += xi_mu2;
    }
    */
    this->sMat = scatterMat;
    scatterMat = (1.0 / (featuresData.rows - 1)) * scatterMat;
    this->covMat = scatterMat;
}
void PCA::calcStats()
{
    calcMeans();
    calcScatterMatrix();
}

void PCA::outputData()
{
    std::cout <<"Data entered is :\n" << featuresData << std::endl;
}
void PCA::outputMu()
{
    
    std::cout <<"Mu's are :\n" << mu << std::endl;
}

void PCA::outputScatterMatrix()
{
    std::cout <<"Scatter Matrix is:\n" << sMat << std::endl;
}
void PCA::outputCovMatrix()
{
    std::cout <<"Cov Matrix is:\n" << covMat << std::endl;
}

void PCA::outputStats()
{
    outputMu();
    outputScatterMatrix();
    outputCovMatrix();
}
Matrix PCA::getScatter()
{
    return Matrix(this->sMat);
}
Matrix PCA::getEigVectors()
{
    return Matrix(this->eigenVect);
}
Matrix PCA::getEigValues()
{
    return Matrix(this->eigenVals);
}
Matrix PCA::getAi()
{
    return Matrix(this->ai);
}
Matrix PCA::getX_bar()
{
    return Matrix(this->x_bar);
}
Matrix PCA::getOrig()
{
    return Matrix(this->featuresData);
}
PCA::~PCA() {};
