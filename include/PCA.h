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
 *  @todo Need to create methods for eigen decomposition. Providing PCA class with eigen value & vector.
 */
#ifndef PCA_PCA_H
#define PCA_PCA_H

#include <vector>
#include <string>



class FeatureData
{

public:
    string name;
    vector<double> data;
    FeatureData(string, vector<double>);

};

/** @warning
 * Do not pass in data with different number of inputs per feature
*/
class PCA
{
private:
    int numFeatures;
    vector<FeatureData>  featuresData;
    vector<double> mu;
    vector<vector<double>> sMat;


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

    vector<vector<double>> vectorMultItTranspose(vector<double>);
    void add2total(vector<vector<double>> &, vector<vector<double>>);

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
