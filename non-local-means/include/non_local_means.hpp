#pragma once
#include "Halide.h"
#include <opencv2/opencv.hpp>
#include <type_traits>

void non_local_means_CUDA(cv::Mat & src, cv::Mat & dest, int search_window, int patch_radius, float stdev_noise, float filter_param_h);
void non_local_means_AVX512(cv::Mat_<cv::Vec3b> &src_, cv::Mat_<cv::Vec3b> &dest, int search_radius, int patch_radius, float stdev_noise, float filter_param_h);

template <typename T>
void non_local_means(cv::Mat_<T> &src, cv::Mat_<T> &dest, int search_radius = 10, int patch_radius = 3)
{
    using Type = typename cv::DataType<T>::channel_type;
    constexpr int channels = cv::DataType<T>::channels;

    const double stdev_noise = 10;
    const double stdev_noise2 = stdev_noise * stdev_noise;
    const double param_h = 10;
    const double param_h2_inv = 1.0 / (param_h * param_h);
    const double k = 1.0 / (channels * (patch_radius * 2 + 1) * (patch_radius * 2 + 1));

    auto reflect = [](int p, int len)
    {
        return p < 0 ? -p : p < len ? p
                                    : len - 1 - (p - len);
    };
#pragma omp parallel for
    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            cv::Vec<double, channels> intensity = 0.0;
            double weight_sum = 0.0;
            for (int ky = -search_radius; ky <= search_radius; ky++)
            {
                for (int kx = -search_radius; kx <= search_radius; kx++)
                {
                    double norm = 0.0;
                    for (int py = -patch_radius; py <= patch_radius; py++)
                    {
                        const int y1 = reflect(y + py, src.rows);
                        const int y2 = reflect(y + ky + py, src.rows);
                        for (int px = -patch_radius; px <= patch_radius; px++)
                        {
                            const int x1 = reflect(x + px, src.cols);
                            const T p1 = src(y1, x1);
                            const int x2 = reflect(x + kx + px, src.cols);
                            const T p2 = src(y2, x2);

                            for (int ch = 0; ch < src.channels(); ch++)
                            {
                                double v = static_cast<double>(p1[ch] - p2[ch]);
                                norm += v * v;
                            }
                        }
                    }
                    double weight = -std::max(norm * k - 2 * stdev_noise2, 0.0) * param_h2_inv;
                    weight = exp(weight);

                    const int y3 = reflect(y + ky, src.rows);
                    const int x3 = reflect(x + kx, src.cols);
                    intensity += src(y3, x3) * weight;
                    weight_sum += weight;
                } // kx
            }     // ky

            const auto val = weight_sum != 0 ? intensity / weight_sum : src(y, x);

            for (int ch = 0; ch < src.channels(); ch++)
            {
                dest(y, x)[ch] = cv::saturate_cast<Type>(val[ch]);
            }
        } // for x
    }     // for y

    return;
}
