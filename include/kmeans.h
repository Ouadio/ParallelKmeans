#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>
#include <time.h>
#include <omp.h>
#include <chrono>
#include <ctime>
#include <limits>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace KM
{

    class KMeans
    {
    private:
        uint16_t m_nb_channels = 3;
        uint16_t m_nb_centroid = 1;
        std::vector<uchar> m_centroids;     //Of size m_nb_centroid*m_nb_channels
        double *m_new_centroids{nullptr};   //Stores temporary sums of pixels to compute barycentre later and evaluate centroid displacement
        uint32_t *m_mapping{nullptr};       //Of size rows * cols, indicates for each flattened index the corresponding centroid (or cluster) [0,0,1,1, etc..]
        uint32_t *m_cluster_sizes{nullptr}; //Of size m_nb_centroid : contains number of pixels for each cluser. Helps computing barycentre with m_new_centroids

    public:
        //Constructor
        KMeans(uint16_t nb_channels, uint16_t nb_centroids);

        //Destructor
        virtual ~KMeans();

        //Setters & Gettersfor centroids
        void setCentroids(std::vector<uchar> centr);
        std::vector<uchar> getCentroids() const;

        //Main kmeans method : iterative computing & update of centroids positions & associated
        //pixels of the image
        int compute_centroids(cv::Mat const &image,
                              int max_iterations,
                              double epsilon,
                              std::string mode);

        //Random centroids initialization based on image dimensions
        void init_centroids(cv::Mat const &image);

        //Updating the centroids based on the surrounding closest pixels,
        //Returns a vector of euclidean differences between old and new position for each centroid

        std::vector<double> update_centroid(cv::Mat const &image,
                                            std::string mode = "seq");

        //Closest centroid
        uint32_t argClosest(double &distance,
                            int row,
                            int col,
                            cv::Mat const &image);

        //Segmentation
        void compute_segmentation(cv::Mat &image,
                                  std::string mode = "seq");

        //Kmeans + Image segmentation
        int process(cv::Mat &image,
                    int max_iterations = 1000,
                    double epsilon = 1,
                    std::string mode = "seq");
    };
} // namespace KM
