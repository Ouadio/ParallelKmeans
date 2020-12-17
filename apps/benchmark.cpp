#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <random>
#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/filesystem.hpp>
#include "kmeans.h"
#include "utils.h"
#include <boost/algorithm/string.hpp>
#include <fstream>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{

    using namespace boost::program_options;
    options_description desc;
    desc.add_options()("help", "produce help")("mode", value<std::string>()->default_value("seq"), "seq | omp | tbb")("k-variations", value<int>()->default_value(4), "Number of k-centroids variations for benchmark");

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 1;
    }

    //BENCHMARKING CASE
    std::cout << "================================================" << std::endl;
    std::cout << "------------------BENCHMARKING------------------" << std::endl;
    std::cout << "================================================" << std::endl;
    std::string samplesDir = "../../test/samples/";
    std::string imageFileBase = samplesDir + "abstract";

    std::string mode = vm["mode"].as<std::string>();

    int nbVariations = vm["k-variations"].as<int>();

    //Hardcoded nb centroids scenarios
    int *nbs_centroids = new int[nbVariations];
    nbs_centroids[0] = 2;
    for (size_t k = 1; k < nbVariations; k++)
    {
        nbs_centroids[k] = nbs_centroids[k-1] * 2;
    }

    //Reading and preparing image & logFile
    Mat imageBench, imageCopy;
    std::ofstream myFile;
    std::string logFile = "logKMeans.csv";
    ifstream iFile;
    iFile.open(logFile);
    myFile.open(logFile, std::ios_base::app);

    if (!iFile)
    {
        myFile << "mode,nRows,nCols,nChannels,nCentroids,nIterations,time,timePerCycle,\n";
    }

    //Looping over image samples (6 for our benchmark in ASCENDING Size)
    for (int i = 0; i <= 5; i++)
    {
        double start = 0;
        double end = 0;

        std::string imageFile = imageFileBase + std::to_string(i) + ".jpg";
        imageBench = imread(imageFile.c_str(), IMREAD_COLOR); // Read the file

        const int channels = imageBench.channels();

        // Check for invalid input
        if (!imageBench.data)
        {
            cout << "Could not open or find the image : " << imageFile << std::endl;
            return -1;
        }

        //Looping over different nb centroids examples : {3, 7, 13, 23}
        for (int k = 0; k < nbVariations; k++)
        {
            imageCopy = imageBench.clone();
            int nb_centroids = nbs_centroids[k];
            //Kmeans
            KM::KMeans algo(channels, nb_centroids);
            start = now();
            int nb_iterations = algo.process(imageCopy, 100, 1, mode);
            end = now();
            //Log
            std::cout << "Overall Time : " << end - start << " | nb centroids :" << nb_centroids << " |  mode : " << mode << std::endl;
            myFile << mode << "," << imageCopy.rows << "," << imageCopy.cols << "," << channels << "," << nb_centroids << "," << nb_iterations << "," << end - start << "," << (end - start) / nb_iterations << ",\n";
        }
    }
    myFile.close();
    std::cout << "\nLog benchmark written at -----------> " << logFile << std::endl;

    delete[] nbs_centroids;
    return 0;
}
