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
Matrix* PCA::eigen()
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
void PCA::eigen(Matrix& V, Matrix& D)
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
    eigenVect = Matrix(sMat.cols, sMat.cols, 0.0);
    eigenVals = Matrix(sMat.cols, sMat.cols, 0.0);
    size_t jiterations, i, j, k, iq, ip, m_size = sMat.cols, jacobi_max_iterations = 1000;
    
    double threshold, theta, tau, t, sm, s, h, g, c,
    p;
    double *b, *z;
    //double b[m_size * sizeof(double)], z[m_size*sizeof(double)];
    
    //Matrix b(m_size, m_size), z(m_size, m_size);
    
    b = new double[m_size];
    b--;
    z = new double[m_size];
    z--;
    
    /* initialize eigenVect and eigen values */
    for( ip = 1; ip < m_size; ip++ ){
        for (iq = 1; iq < m_size; iq++){
            eigenVect(ip,iq) = 0.0;
        }
        eigenVect(ip,ip) = 1.0;
    }
    for( ip = 1; ip < m_size; ip++ ){
        b[ip] = eigenVals(ip,ip) = sMat(ip,ip);
        z[ip] = 0.0;
    }
    
    jiterations = 0;
    for( i = 0; i <= jacobi_max_iterations; i++ ){
        sm = 0.0;
        for( ip = 1; ip < m_size; ip++ ){
            for( iq = ip + 1; iq < m_size; iq++ ){
                sm += fabs(sMat(ip,iq));
            }
        }
        
        if( sm == 0.0 ){
            /* eigenVals & eigenVect sorting */
            for( i = 1; i < m_size; i++ ){
                p = eigenVals(k = i, i);
                for( j = i + 1; j < m_size; j++ ){
                    if( eigenVals(j, j) >= p ){
                        p = eigenVals(j,k = j);
                    }
                }
                if( k != i ){
                    eigenVals(0,k) = eigenVals(0,i);
                    eigenVals(0,i) = p;
                    for( j = 1; j < m_size; j++ ){
                        p = eigenVect(j, i);
                        eigenVect(j, i) = eigenVect(j, k);
                        eigenVect(j, k) = p;
                    }
                }
            }
            
            /* restore symmetric sMat's sMat */
            for( i = 2; i < m_size; i++ ){
                for( j = 1; j < i; j++ ){
                    sMat(j,i) = sMat(i,j);
                }
            }
            
            z++;
            delete z;
            b++;
            delete b;
            jit =  jiterations;
            return;
        }
        
        
        threshold = ( i < 4 ? 0.2 * sm / (m_size * m_size) : 0.0 );
        for( ip = 1; ip < m_size; ip++ )
        {
            for( iq = ip + 1; iq < m_size; iq++ )
            {
                g = 100.0 * fabs(sMat(ip,iq));
                
                if( i > 4 &&
                   fabs(eigenVals(ip, ip)) + g == fabs(eigenVals(ip, ip)) &&
                   fabs(eigenVals(iq, iq)) + g == fabs(eigenVals(iq, iq)) ){
                    sMat(ip,iq) = 0.0;
                }
                else if( fabs(sMat(ip,iq)) > threshold ){
                    h = eigenVals(iq, iq) - eigenVals(ip, ip);
                    if( fabs(h) + g == fabs(h) ){
                        t = sMat(ip,iq) / h;
                    }
                    else {
                        theta = 0.5 * h / sMat(ip,iq);
                        t     = 1.0 / ( fabs(theta) + sqrt( 1.0 + theta * theta ) );
                        if( theta < 0.0 ){
                            t = -t;
                        }
                    }
                    
                    c                = 1.0 / sqrt(1 + t * t);
                    s                = t * c;
                    tau              = s / (1.0 + c);
                    h                = t * sMat(ip,iq);
                    z[ip]           -= h;
                    z[iq]           += h;
                    eigenVals(ip, ip) -= h;
                    eigenVals(iq, iq) += h;
                    sMat(ip, iq)   = 0.0;
                    
                
                    #define M_ROTATE(M,i,j,k,l) g = M(i,j); \
                    h = M(k,l); \
                    M(i,j) = g - s * (h + g * tau); \
                    M(k,l) = h + s * (g - h * tau)
                    for( j = 1; j < ip; j++ ){
                        M_ROTATE( sMat, j, ip, j, iq );
                    }
                    for( j = ip + 1; j < iq; j++ ){
                        M_ROTATE( sMat, ip, j, j, iq );
                    }
                    for( j = iq + 1; j < m_size; j++ ){
                        M_ROTATE( sMat, ip, j, iq, j );
                    }
                    for( j = 1; j < m_size; j++ ){
                        M_ROTATE( eigenVect, j, ip, j, iq );
                    }
                    
                    ++jiterations;
                }
            }
        }
        
        /*for( ip = 1; ip < m_size; ip++ ){
         b[ip)          += z[ip);
         eigenVals[ip) = b[ip);
         z[ip)           = 0.0;
         }*/
    }
    
    z++;
    delete z;
    b++;
    delete b;
    
}

void PCA::houseHolder(Matrix& A, Matrix& d, Matrix& e, const size_t& row)
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


/**@field:
  *@bug: friend operator wont take matrix as const matrix& unless it is const
 */

void PCA::calcScatterMatrix()
{
    Matrix scatterMat(mu.rows, mu.rows, 0.0);
    //Matrix xi;
    //Matrix xi_mu;
    for(size_t c = 0; c < featuresData.cols; c++)
    {
        // calc all xi - mu
        const Matrix xi = featuresData.getColumn(c);
        //xi = featuresData.getColumn(c);
        //xi_mu = xi - mu;
        const Matrix xi_mu = xi - mu;
        // calc sum((xi-mu) * (xi-mu))
        Matrix xi_mu2 = xi_mu * ~(xi_mu);
        scatterMat += xi_mu2;
    }
    /*
    // calc all xi-mu
    vector<vector<double>> x_mus; // all x-mu for all features
    vector<double> featDif; // all x-mu for individual features
    difs.clear();
    int featNum = 0;
    for(auto feature : this->featuresData)
    {

        featDif.clear();
        for(auto input : feature.data)
        {
            double x_mu = input - mu[i];
            featDif.push_back(x_mu);
        }
        x_mus.push_back(featDif);
    }

    // Calc each (xi-mu)*(xi-mu)'
    vector<vector<double>> scatterMat;
    scatterMat.clear();
    for(int i = 0; i < this->featuresData.data.size(); i++)
    {



        // gather all one input for all features
        vector<double> inputN;
        inputN.clear();
        int input = 0;
        for (auto feature : x_mus) {
            double sample = feature[input];
            input++;
            inputN.push(back);
        }
        // get variance for that input
        vector<vector<double>> cScatterMat = vectorMultItTranspose(inputN);
        add2total(scatterMat, cScatterMat);
    }
     */
    this->sMat = scatterMat;
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

void PCA::outputStats()
{
    outputMu();
    outputScatterMatrix();
}
Matrix PCA::getScatter()
{
    return Matrix(this->sMat);
}
PCA::~PCA() {};
