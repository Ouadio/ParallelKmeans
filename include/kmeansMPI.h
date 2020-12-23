#pragma once

#include <cstdint>
#include <vector>
#include <cuchar>
#include <opencv2/opencv.hpp>

namespace KMMPI
{
    //Random centroids initialization based on image dimensions
    void init_centroids(cv::Mat const &image, int nb_channels, int nb_centroid, std::vector<uchar> &centroids);

    //Closest centroid
    uint32_t argClosest(double &distance, int index, std::vector<u_char> const &flat_image, int nb_channels, int nb_centroids, std::vector<u_char> centroids);

    //Updating centroid based on pixels proximity (partial operation since we don't have the whole image)
    void update_centroid(std::vector<u_char> const &flat_image,
                         int total_length,
                         int nb_centroid,
                         int nb_channels,
                         std::vector<u_char> centroids,
                         std::vector<long> &cluster_sizes,
                         std::vector<float> &newCentroids,
                         std::vector<uint16_t> &mapping);

    //Segmentation
    void gen_segmentation(std::vector<u_char> &flat_image,
                          int nb_channels,
                          int total_length,
                          std::vector<u_char> const &centroids,
                          std::vector<uint16_t> const &mapping);
} // namespace KMMPI