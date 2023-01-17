#include "Halide.h"
#include "halide_image_io.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <type_traits>
#include "non_local_means.hpp"

int main(int argc, char **argv)
{
    cv::Mat a = cv::imread("./inu.jpg", cv::IMREAD_UNCHANGED);
    cv::Mat_<cv::Vec3b> img = a.clone();
    cv::TickMeter timer;
    // AVX512
    cv::Mat_<cv::Vec3b> dest_AVX = cv::Mat_<cv::Vec3b>::zeros(img.size());
    timer.reset();
    timer.start();
    non_local_means_AVX512(img, dest_AVX, 5, 2, 10.f, 10.f);
    timer.stop();
    std::cout << timer.getTimeSec() << " [s]" << std::endl;

    // Cuda
    cv::Mat_<cv::Vec3b> dest_cuda = cv::Mat_<cv::Vec3b>::zeros(img.size());
    non_local_means_CUDA(img, dest_cuda, 5, 2, 10.f, 10.f);

    // naive
    cv::Mat_<cv::Vec3b> dest = cv::Mat_<cv::Vec3b>::zeros(img.size());
    timer.reset();
    timer.start();
    non_local_means(img, dest, 5, 2);
    timer.stop();
    std::cout << timer.getTimeSec() << " [s]" << std::endl;

    // Halide
    Halide::Buffer<uint8_t> src = Halide::Tools::load_image("./inu.jpg");

    timer.reset();
    timer.start();
    Halide::Func out = non_local_means_Halide(src, 5, 2);
    out.compile_jit(Halide::get_jit_target_from_environment());
    timer.stop();
    std::cout << timer.getTimeSec() << " [s]" << std::endl;
    timer.reset();
    timer.start();
    Halide::Buffer<uint8_t> h = out.realize({src.width(), src.height(), 3});

    timer.stop();
    std::cout << timer.getTimeSec() << " [s]" << std::endl;

    cv::Mat_<cv::Vec3b> dest_h(cv::Size(src.width(), src.height()));
    for (int y = 0; y < src.height(); y++)
    {
        for (int x = 0; x < src.width(); x++)
        {
            for (int c = 0; c < src.channels(); c++)
            {
                dest_h(y, x)[c] = h(x, y, c);
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