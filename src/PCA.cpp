//
// Created by Ariel Feliz on 11/26/16.
//
#include "include/PCA.h"
#include <cstdarg.h> //va_list, va_start, va_end
#include <assert.h>
FeatureData::FeatureData(string name, vector<double> data): name(name), data(data) {}


PCA::PCA(): numFeatures(0)
{
    featuresData.clear();
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

void PCA::calcMus()///O(numFeatures*numInputsPerFeature)
{

    for(auto it : this->featuresData) //for(auto it = this->inputData.begin(); it != this->inputData.end(); ++it)
    {
        double sum = 0;
        for(it2 : it.data)
        {
            sum += it2;
        }
        double avg = sum/it.data.size();
        this->mu.push_back(avg);
    }
}
/*
 * one xi-mu for all features
 */
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
    /*
    for(int from = 0, to = 0; from < total.size() && to < total.size();from++, to++)
    {
        for(int from = 0, to = 0; from < total.size() && to < total.size();from++, to++)
    }

    */
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

void PCA::calcScatterMatrix()
{
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
    this->sMat = scatterMat;
}





