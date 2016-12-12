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
PCA::~PCA() {};
