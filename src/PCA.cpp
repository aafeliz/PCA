#if 0
/** @file PCA.cpp
 *  @brief Principal Component Analysis Class.
 *
 *  Contains class to calculate most sinigicant feature for a given class
 *  and inputs for those features.
 *
 *  @author Ariel Feliz(aafeliz)
 *  @author Elvin (<githubid>)
 *  @date 11/26/16
 *  @bug No known bugs.
 *  @todo Need to create methods for eigen decomposition. Providing PCA class with eigen value & vector.
 */
#include "PCA.h"
//#include <cstdarg.h> //va_list, va_start, va_end
#include <assert.h>
FeatureData::FeatureData(string name, Matrix data): FeatureName(name), data(data) {}


PCA::PCA(): numFeatures(0)
{
    featuresData.clear();
}


/**@brief: constructor for PCA when features and data are being passed in
 */
PCA::PCA(const Matrix& featuresData)
{
    this->featuresData = featuresData;
}
void PCA::passFeaturesData(int numFeatures, ...)
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
}
void PCA::passFeaturesData(vector<FeatureData> features)
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
}
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
    for(int r = 0; r < featuresData.rows; r++)
    {
        double featSum = 0;
        for(int c = 0; c < featuresData.cols; c++)
            featSum += featuresData(r, c);
        double avg = featSum/((double)featuresData.cols);
        arr[r] = avg;

    }
    //Matrix m(numfeatures, 1, arr);
    mu = Matrix(numfeatures, 1, arr);
}

/*
/**@brief: one xi-mu for all features
 */
/*
vector<vector<double>> PCA::vectorMultItTranspose(vector<double> featsX_mu)
{
    vector<vector<double>> matrix(featsX_mu.size());
    for(int i = 0; i < featsX_mu.size(); i++)
    {
        matrix[i].reserve(featsX_mu.size());
    }

    //int i = 0;
    for(int col = 0; col < featsX_mu.size(); col++)
    {
        //int j = 0;
        for(int row = 0; row < featsX_mu.size(); row++)
        {
            matrix[col][row] = featsX_mu[row] * featsX_mu[col];

           //j++;
        }
        //i++;
    }
    return matrix;
}
void PCA::add2total(vector<vector<double>> &total, vector<vector<double>> current)
{

    if(total.empty())
    {
        total = current;
        return;
    }
    for(auto from = total.begin(), to = current.begin(); from != total.end() && to != total.end();++from, ++to)
    {
        for(auto fItem = *from.begin(), tItem = *to.begin(); fItem != *from.end() && tItem != *to.end();++fItem, ++tItem)
        {
            *fItem += *tItem;
        }
    }
}
*/

void PCA::calcScatterMatrix()
{
    Matrix scatterMat(mu.rows, mu.rows, 0);
    for(size_t c = 0; c < featuresData.cols; c++)
    {
        // calc all xi - mu
        Matrix xi_mu = featuresData.getColumn(c) - mu;
        // calc sum((xi-mu) * (xi-mu))
        Matrix xi_mu2 = xi-mu * ~xi-mu;
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



#endif

