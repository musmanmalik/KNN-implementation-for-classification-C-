// projectKmeans.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
using namespace std;

#ifndef _FILE_NAME
#define _FILE_NAME "datasetwithoutput2.txt"
#endif 

double** get2DArray(const int rows, const int cols);

int countNumberofLines(const char* filename);

double** getData(const char* filename, int*& labels, int& rows, int& cols);

void randomShuffleData(double** features, int* labels, const int rows, const int cols, const int shuffles);

int addDistance(double* k_distances, int* k_labels, double dist, int label, int current_size, const int K);

double euclideanDistance(double* point1, double* point2, const int cols);

int mostFrequent(int* labels, int size);

void KNN(double** features, double** testFeatures, int* labels, int* testLabels, const int rowsTest, const int rowsTrain, const int cols, const int K);

int main(int argc, char** argv)
{

    cout << "Enter the value of K: ";
    int K = 0;
    cin >> K;

    int rows, cols, * labels;

    // get labels and data rows from the file
    double** features = getData(_FILE_NAME, labels, rows, cols);


    // randomly shuffle (1/3) of all the rows
    randomShuffleData(features, labels, rows, cols, rows / 3);

    /*for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            cout << features[i][j] << " ";
        }
        cout << labels[i];
        cout << endl;
    }*/


    // test data is 20%
    int rowsTest = rows * 0.2;

    // rest is taining dataa
    int rowsTrain = rows - rowsTest;

    double** testFeatures = &features[rowsTrain];
    int* testLabels = &labels[rowsTrain];

    /*for (int i = 0; i < rowsTest; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            cout << testFeatures[i][j] << " ";
        }
        cout << testLabels[i];
        cout << endl;
    }*/


    

    // run serial KNN
    KNN(features, testFeatures, labels, testLabels, rowsTest, rowsTrain, cols, K);

    delete[] features[0]; delete[] features;
    delete[] labels;

    return 0;
}


// create a 2D array
double** get2DArray(const int rows, const int cols)
{
    // allocate row pointers
    double** arr = new double* [rows];

    // allocate dynamic memory
    double* mem = new double[rows * cols];

    // rearrange row pointers
    for (int i = 0; i < rows; i++)
    {
        arr[i] = &mem[i * cols];
    }
    return arr;
}


int countNumberofLines(const char* filename)
{
    ifstream fin(filename);

    if (!fin.is_open())
    {
        return -1;
    }

    string line = "";
    int count = 0;

    // read lines unless file is ended
    while (getline(fin, line))
    {
        count++;
    }

    fin.close();

    return count;
}




double** getData(const char* filename, int*& labels, int& rows, int& cols)
{
    ifstream fin(filename);


    // if file doesn't exist then no data is retreived
    if (!fin.is_open())
    {
        return NULL;
    }


    // count number of lines in the file
    rows = countNumberofLines(filename);

    // use the firstline to count number of columns
    string firstLine = "";
    getline(fin, firstLine);
    stringstream ssFirst(firstLine);
    string word = "";

    cols = 0;
    while (ssFirst >> word)
    {
        ++cols;
    }

    // last one is the label
    cols--;

    double** features = get2DArray(rows, cols);
    labels = new int[rows];

    int i = 0;

    // read all the data from the file
    string line = firstLine;
    do
    {
        stringstream ss(line);
        int j = 0;

        // read all the columns of a row
        while (j < cols)
        {
            ss >> features[i][j];
            j++;
        }

        // read the current label
        ss >> labels[i];
        i++;
    } while (getline(fin, line));

    fin.close();

    return features;


}

void randomShuffleData(double** features, int* labels, const int rows, const int cols, const int shuffles)
{
    double* tempRow = new double[cols];
    for (int i = 0; i < shuffles; i++)
    {
        // randomly shuffle two rows
        int r1 = rand() % rows;
        int r2 = rand() % rows;


        if (r1 != r2)
        {


            //swap the data rows
            for (int j = 0; j  < cols; j++)
            {
                tempRow[j] = features[r1][j];
                features[r1][j] = features[r2][j];
                features[r2][j] = tempRow[j];
            }

            int temp = labels[r1];
            labels[r1] = labels[r2];
            labels[r2] = temp;
        }
        else
            i--;

    }

    delete[] tempRow;
}

int addDistance(double* k_distances, int* k_labels, double dist, int label, int current_size, const int K)
{
    if (current_size == 0)
    {
        k_distances[0] = dist;
        k_labels[0] = label;
    }
    else
    {
        int i = current_size - 1;

        if (current_size < K || k_distances[i] > dist)
        {
            // start from end and find the exact position of add
            // so that array stays sorted
            while (i >= 0 and k_distances[i] > dist)
            {
                // if already full then last element will be removed
                if (i != K - 1)
                {
                    k_distances[i + 1] = k_distances[i];
                    k_labels[i + 1] = k_labels[i];
                }
                i -= 1;
            }
            k_distances[i + 1] = dist;
            k_labels[i + 1] = label;
        }
    }

    if (current_size == K)
        return current_size;
    return current_size + 1;
}

double euclideanDistance(double* point1, double* point2, const int cols)
{
    // eulidean distance between given points
    double distance = 0;
    for (int j = 0; j < cols; j++)
    {
        double val = point1[j] - point2[j];
        distance += (val * val);
    }

    return sqrt(distance);
}


// Moore's voting Algorithm
int mostFrequent(int* labels, int size) {
    int res = 0;
    int count = 1;
    for (int i = 1; i < size; i++)
    {
        if (labels[i] == labels[res]) {
            count++;
        }
        else
        {
            count--;
        }

        if (count == 0)
        {
            res = i; count = 1;
        }

    }
    return labels[res];
}

void KNN(double** features, double** testFeatures, int* labels, int* testLabels, const int rowsTest, const int rowsTrain, const int cols, const int K)
{
    int accurate = 0;


    // take a test example
    for (int i = 0; i < rowsTest; i++)
    {
        double* k_distances = new double[K];
        int* k_labels = new int[K];
        int current_size = 0;

        // find K closest neightbours using train data
        for (int j = 0; j < rowsTrain; j++)
        {
            double dist = euclideanDistance(&testFeatures[i][0], &features[j][0], cols);
            current_size = addDistance(k_distances, k_labels, dist, labels[j], current_size, K);
        }

        // predict the label
        int predictedLabel = mostFrequent(k_labels, K);

        if (predictedLabel == testLabels[i])
            accurate += 1;
    }


    cout << "Accuracy obtained (Serial) is: " << (accurate / (double)rowsTest) * 100.0 << endl;
}