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
#include <boost/algorithm/string.hpp>
#include <fstream>
#include "sys/time.h"
#include <ctime>

using namespace cv;
using namespace std;

double now()
{
  struct timeval t;
  double f_t;
  gettimeofday(&t, NULL);
  f_t = t.tv_usec;
  f_t = f_t / ((float)1E6);
  f_t += t.tv_sec;
  return (f_t);
}

int main(int argc, char **argv)
{

  using namespace boost::program_options;
  options_description desc;
  desc.add_options()("help", "produce help")("file", value<std::string>(), "image file")("show", value<int>()->default_value(0), "show image")("kmean-value", value<int>()->default_value(3), "KMean k value")("rgb", value<int>()->default_value(1), "RGB or GrayScale")("mode", value<std::string>(), "seq | omp")("benchmark", value<int>()->default_value(0), "Generate benchmark");

  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << "\n";
    return 1;
  }

  //BENCHMARKING CASE
  if (vm["benchmark"].as<int>() == 1)
  {

    std::cout << "================================================" << std::endl;
    std::cout << "------------------BENCHMARKING------------------" << std::endl;
    std::cout << "================================================" << std::endl;
    std::string samplesDir = "../test/samples/";
    std::string imageFileBase = samplesDir + "abstract";

    std::string mode = vm["mode"].as<std::string>();

    //Hardcoded nb centroids scenarios
    int nbs_centroids[4] = {3, 7, 13, 23};

    //Reading and preparing image & logFile
    Mat imageBench, imageCopy;
    std::ofstream myFile;
    std::string logFile = "./logs/logKMeans.csv";
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
      for (int k = 0; k < 4; k++)
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
    return 0;
  }

  std::string img_file = vm["file"].as<std::string>();
  Mat image;

  if (vm["rgb"].as<int>() == 1)
  {
    image = imread(img_file.c_str(), IMREAD_COLOR); // Read the file
  }
  else
  {
    image = imread(img_file.c_str(), IMREAD_GRAYSCALE); // Read the file
  }

  if (!image.data) // Check for invalid input
  {
    cout << "Could not open or find the image" << std::endl;
    return -1;
  }

  cout << "NB CHANNELS : " << image.channels() << std::endl;
  cout << "NROWS       : " << image.rows << std::endl;
  cout << "NCOLS       : " << image.cols << std::endl;
  const int channels = image.channels();
  std::string mode = vm["mode"].as<std::string>();
  std::string modeDispl = mode;
  boost::to_upper(modeDispl);

  double start, end;

  int nb_centroids = vm["kmean-value"].as<int>();

  if (channels == 3)
  {
    cout << "========= RGB Kmeans | MODE : " << modeDispl << " | NB Clusters : " << nb_centroids << "  =========" << endl;
  }
  else
  {
    cout << "========= GrayScale Kmeans | MODE : " << modeDispl << " | NB Clusters : " << nb_centroids << "  =========" << endl;
  }
  KM::KMeans algo(channels, nb_centroids);

  start = now();
  int nb_iterations = algo.process(image, 100, 1, mode);
  end = now();

  std::cout << "Time : " << end - start << std::endl;
  std::cout << "Average Time per cycle : " << (end - start) / nb_iterations << std::endl;

  img_file.resize(img_file.size() - 4);
  string fileName = img_file + "_" + mode + "_Segmented.jpg";
  imwrite(fileName, image);
  cout << "Image written at : " << fileName << endl;

  return 0;
}
