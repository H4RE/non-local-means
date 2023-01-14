#include "cuda_runtime.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "non_local_means.hpp"

__global__ void non_local_means_CUDA_(unsigned char *src, unsigned char *dest, int width, int height, int channels, int search_radius, int patch_radius, float stdev_noise, float filter_param_h)
{

    const int patch_sizeXch = (patch_radius * 2 + 1) * (patch_radius * 2 + 1) * channels;

    const float stdev_noise2 = stdev_noise*stdev_noise;
    const float filter_param_h2 = filter_param_h * filter_param_h;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int offset = patch_radius + search_radius;
    const int d_width= width-2*offset;
    if (offset < x && x < width - offset && offset < y && y < height - offset)
    {
        float intensity[3] = {};
        double weight_sum = 0.0;
        for (int ky = -search_radius; ky <= search_radius; ky++)
        {
            for (int kx = -search_radius; kx <= search_radius; kx++)
            {
                double norm = 0.0;
                for (int py = -patch_radius; py <= patch_radius; py++)
                {
                    const int y1 = y + py;
                    const int y2 = y + ky + py;
                    for (int px = -patch_radius; px <= patch_radius; px++)
                    {
                        for (int ch = 0; ch < channels; ch++)
                        {
                            const int x1 = x + px;
                            const uchar p1 = src[y1 * width * channels + x1 * channels + ch];
                            const int x2 = x + kx + px;
                            const uchar p2 = src[y2 * width * channels + x2 * channels + ch];

                            double v = static_cast<double>(p1 - p2);
                            norm += v * v;
                        }
                    }
                }
                double weight = -max(norm /patch_sizeXch- 2 * stdev_noise2, 0.0) / filter_param_h2;
                weight = exp(weight);
                weight_sum += weight;

                for (int ch = 0; ch < channels; ch++)
                {
                    intensity[ch] += src[(y + ky) * width * channels + (x + kx) * channels + ch] * weight;
                }
            } // kx
        }     // ky

        for (int ch = 0; ch < channels; ch++)
        {
            double value = intensity[ch] / weight_sum;
            int round = (int)(value + (value >= 0 ? 0.5 : -0.5));
            uchar sc = static_cast<uchar>(min(value, (double)UCHAR_MAX));

            dest[(y-offset) *d_width * channels + (x-offset) * channels + ch] = sc;
        }
    }
}
void non_local_means_CUDA(cv::Mat &src, cv::Mat &dest, int search_radius, int patch_radius, float stdev_noise, float filter_param_h)
{
    cudaEvent_t start, end;
    cudaEventCreate(&start);
	cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    cv::Mat src_ex =src.clone();
    cv::copyMakeBorder(src_ex, src_ex, search_radius + patch_radius, search_radius + patch_radius, search_radius + patch_radius, search_radius + patch_radius, cv::BORDER_REFLECT101);
    uchar *dev_src = nullptr;
    uchar *dev_dest = nullptr;

    cudaMalloc(&dev_src, src_ex.total() * sizeof(uchar) * src_ex.channels());
    cudaMemcpy(dev_src, src_ex.data, src_ex.total() * sizeof(uchar) * src_ex.channels(), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_dest, dest.total() * sizeof(uchar) * dest.channels());
    cudaMemcpy(dev_dest, dest.data, dest.total() * sizeof(uchar) * dest.channels(), cudaMemcpyHostToDevice);

    dim3 block(32, 32); // 1024
    dim3 grid((src_ex.cols + block.x - 1) / block.x, (src_ex.rows + block.y - 1) / block.y);

    non_local_means_CUDA_<<<grid, block>>>(dev_src, dev_dest, src_ex.cols, src_ex.rows, src.channels(), search_radius, patch_radius, stdev_noise, filter_param_h);
    cudaDeviceSynchronize();

    cv::Mat mat = cv::Mat::zeros(dest.size(), dest.type());
    cudaMemcpy(mat.data, dev_dest, dest.total() * sizeof(uchar) * dest.channels(), cudaMemcpyDeviceToHost);
    mat.copyTo(dest);

    cudaFree(dev_src);
    cudaFree(dev_dest);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    std::cout << time_ms/1000 <<" [s]"<<std::endl;

}