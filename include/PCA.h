/** @file PCA.h
 *  @brief Principal Component Analysis Class.
 *
 *  Contains class to calculate most sinigicant feature for a given class
 *  and inputs for those features.
 *
 *  @author Ariel Feliz(aafeliz)
 *  @author Elvin Abreu(elvinabreu)
 *  @date 11/26/16
 *  @bug No known bugs.
 *  @todo reimplement methods using support methods from matrix
 */
#ifndef PCA_PCA_H
#define PCA_PCA_H

#include <vector>
#include <string>
#include "Matrix.h"

/*class FeatureData
{
public:
    std::string featureName;
    Matrix data;
    FeatureData(std::string, Matrix);

};*/

/** @warning
 * Do not pass in data with different number of inputs per feature
*/
class PCA
{
private:
    size_t numFeatures;
    Matrix featuresData;
    /**@brief: this can be passed in to know if rows/cols are input nums or features
     * @todo: can be done after matrix class is of type template
     */
    Matrix names;//

    Matrix mu; // more like a vector so it [numFeatures(rows) x 1(cols)]
    Matrix sMat; // scatter matrix
    Matrix A;
    size_t jit;


    /**@brief
     * calculate each features mean and append to mu
     */
    void calcMeans();
    /**@brief
     * calculate xi - mu
     */

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
    /**@brief: constructor for PCA when features and data are being passed in
     */
    PCA(const Matrix& featuresData);
    /**@brief: deconstructor
     */
    ~PCA();
    /**@brief: copy constructor
     */
    PCA(const PCA&);
    /**@brief: move constructor
     */
    PCA(PCA&&);
    
    /**@todo:
     */
    // diganoal everything else zero
    Matrix eye(const Matrix& orig);
    Matrix minor(const Matrix& z, const Matrix& k);
    Matrix* eigen();
    void eigen(Matrix& Q, Matrix& R);
    void eigenJacobian();
    
    Matrix eigenVals; // matrix of eigen values
    Matrix eigenVect; // matrix of eigen vectors
    
    void houseHolder(Matrix& A, Matrix& d, Matrix& e, const size_t& row);
    void ql(Matrix& A, Matrix& d, Matrix& e, const size_t& row);
    Matrix makeD(const Matrix& d, const Matrix& e, const size_t& r);
    
    
    // will return array as [Q, R]
    Matrix* qrDecomp(const Matrix& A);
    // will change Q and R passed in
    void qrDecomp(const Matrix& A, Matrix& Q, Matrix& R);
    
    
    /**@brief
     * two different to pass in input data
     * 1- pass in one feature, or multiple features at a time.
     * 2- pass in all features at once.
     */
    void passFeaturesData(int numFeatures, ...); // expecting arguments of type FeatureData
   // void passFeaturesData(vector<FeatureData> features);
    void passFeaturesData(Matrix features);
    /*
     * calculate means and scatter matrix
     */
    
    void calcStats();
    
    void outputData();
    
    void outputMu();
    
    void outputScatterMatrix();
    
    void outputStats();
    
    Matrix getScatter();


};
#endif //PCA_PCA_H
