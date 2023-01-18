#include "Halide.h"
#include "halide_image_io.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <type_traits>
#include "non_local_means.hpp"
#include "non_local_means_halide.h"

int main(int argc, char **argv)
{
    cv::Mat a = cv::imread("./inu.jpg", cv::IMREAD_UNCHANGED);
    cv::Mat_<cv::Vec3b> img = a.clone();
    cv::TickMeter timer;
    // AVX512
    cv::Mat_<cv::Vec3b> dest_AVX = cv::Mat_<cv::Vec3b>::zeros(img.size());
    {
        timer.reset();
        timer.start();
        non_local_means_AVX512(img, dest_AVX, 5, 2, 10.f, 10.f);
        timer.stop();
        std::cout << "AVX:" << timer.getTimeSec() << " [s]" << std::endl;
    }

    // Cuda
    cv::Mat_<cv::Vec3b> dest_cuda = cv::Mat_<cv::Vec3b>::zeros(img.size());
    std::cout << "CUDA:";
    non_local_means_CUDA(img, dest_cuda, 5, 2, 10.f, 10.f);

    // naive
    cv::Mat_<cv::Vec3b> dest = cv::Mat_<cv::Vec3b>::zeros(img.size());
    timer.reset();
    timer.start();
    non_local_means(img, dest, 5, 2);
    timer.stop();
    std::cout << "Naive:" << timer.getTimeSec() << " [s]" << std::endl;

    // Halide
    Halide::Runtime::Buffer<uint8_t, 3> src = Halide::Tools::load_image("./inu.jpg");
    Halide::Runtime::Buffer<uint8_t, 3> dest_hb(src.width(), src.height(), src.channels());
    timer.reset();
    timer.start();
    non_local_means_halide(src, 5, 2, 10.f, 10.f, dest_hb);
    timer.stop();
    std::cout <<"Halide:" << timer.getTimeSec() << " [s]" << std::endl;
    cv::Mat_<cv::Vec3b> dest_h(cv::Size(src.width(), src.height()));
    {

        for (int y = 0; y < src.height(); y++)
        {
            for (int x = 0; x < src.width(); x++)
            {
                for (int c = 0; c < src.channels(); c++)
                {
                    dest_h(y, x)[c] = dest_hb(x, y, c);
                }
            }
        }
        if (dest_h.channels() == 3)
        {
            std::vector<cv::Mat> s;
            cv::split(dest_h, s);
            std::swap(s[0], s[2]); // OpenCV BGR
            cv::merge(s, dest_h);
        }
    }

    cv::imshow("img", img);
    cv::imshow("naive", dest);
    cv::imshow("AVX", dest_AVX);
    cv::imshow("halide", dest_h);
    cv::imshow("cuda", dest_cuda);
    cv::imshow("diff cuda - halide", (dest_cuda - dest_h) * 5);
    cv::imshow("diff naive - halide", (dest - dest_h) * 5);
    cv::imshow("diff AVX - halide", (dest_AVX - dest_h) * 5);
    cv::waitKey();
    return 0;
}