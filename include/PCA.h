/** @file PCA.h
 *  @brief Principal Component Analysis Class.
 *
 *  Contains class to calculate most sinigicant feature for a given class
 *  and inputs for those features.
 *
 *  @author Ariel Feliz(aafeliz)
 *  @author Elvin (<githubid>)
 *  @date 11/26/16
 *  @bug No known bugs.
 *  @todo reimplement methods using support methods from matrix
 */
#ifndef PCA_PCA_H
#define PCA_PCA_H

#include <vector>
#include <string>



class FeatureData
{

public:
    string featureName;
    Matrix data;
    FeatureData(string, Matrix);

};

/** @warning
 * Do not pass in data with different number of inputs per feature
*/
class PCA
{
private:
    int numFeatures;
    Matrix featuresData;
    /**@brief: this can be passed in to know if rows/cols are input nums or features
     * @todo: can be done after matrix class is of type template
     */
    Matrix names;//

    Matrix mu; // more like a vector so it [numFeatures(rows) x 1(cols)]
    Matrix sMat; // scatter matrix
    Matrix eigenVals; // matrix of eigen values
    Matrix eigenVect; // matrix of eigen vectors


    /**@brief
     * calculate each features mean and append to mu
     */
    void calcMeans();
    /**@brief
     * calculate scatter matrix
     * which will contain covariance and variance
     * sMat = sum((xi - mu)(xi - mu)')
     * where xi is each input, that is correlated to the feature of that mu
     */
    void calcScatterMatrix();
    void calcEigenDecomp();

public:
    PCA();
    /**@brief
     * two different to pass in input data
     * 1- pass in one feature, or multiple features at a time.
     * 2- pass in all features at once.
     */
    void passFeaturesData(int numFeatures, ...); // expecting arguments of type FeatureData
    void passFeaturesData(vector<FeatureData> features);
    void passFeaturesData(Matrix features);
    /*
     * calculate means and scatter matrix
     */
    void calcStats();


};
#endif //PCA_PCA_H
