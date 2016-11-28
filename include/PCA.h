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
    FeatureData(string, vector<double>);

};

/** @warning
 * Do not pass in data with different number of inputs per feature
*/
class PCA
{
private:
    int numFeatures;
    Matrix  featuresData;
    Matrix mu; // more like a vector so it [1 x numFeatures]
    Matrix sMat;


    /**@brief
     * calculate each features mean and append to mu
     */
    void calcMus();
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
    /*
     * calculate means and scatter matrix
     */
    void calcStats();


};
#endif //PCA_PCA_H
