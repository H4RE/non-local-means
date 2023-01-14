#include "Halide.h"
#include "halide_image_io.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <type_traits>
#include "non_local_means.hpp"

template <typename T>
void non_local_means(cv::Mat_<T> &src, cv::Mat_<T> &dest, int search_radius = 10, int patch_radius = 3)
{
    using Type = cv::DataType<T>::channel_type;
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

Halide::Func non_local_means_Halide(Halide::Buffer<uint8_t> &src, int search_radius, int patch_radius)
{
    Halide::Var x, y, c;
    Halide::Var dx, dy;

    Halide::Func src_ex = Halide::BoundaryConditions::mirror_image(src);
    Halide::Func srcf_ex;
    srcf_ex(x, y, c) = Halide::cast<float>(src_ex(x, y, c));

    Halide::Func norm;
    norm(x, y, dx, dy, c) = Halide::pow(src_ex(x, y, c) - srcf_ex(x + dx, y + dy, c), 2);
    Halide::RDom ch(0, 3);
    Halide::Func norm_color;
    norm_color(x, y, dx, dy) += norm(x, y, dx, dy, ch);
    Halide::RDom patch(-patch_radius, patch_radius * 2 + 1, -patch_radius, patch_radius * 2 + 1);
    Halide::Func norm_sum;
    norm_sum(x, y, dx, dy) = Halide::sum(norm_color(x + patch.x, y + patch.y, dx, dy));
    Halide::Func weight;

    const float stdev_noise2 = 10.0 * 10.0;
    const float filter_param_h2 = 10.0 * 10.0;
    const int patch_sizeXch = (patch_radius * 2 + 1) * (patch_radius * 2 + 1) * 3;

    weight(x, y, dx, dy) = Halide::exp(-Halide::max(norm_sum(x, y, dx, dy) / (float)patch_sizeXch - 2*stdev_noise2, 0) / filter_param_h2);
    Halide::RDom swin(-search_radius, search_radius * 2 + 1, -search_radius, search_radius * 2 + 1);
    Halide::Func weight_sum;
    weight_sum(x, y) += weight(x, y, swin.x, swin.y);
    Halide::Func intensity;
    intensity(x, y, c) += weight(x, y, swin.x, swin.y) * srcf_ex(x + swin.x, y + swin.y, c);

    Halide::Func dest;
    dest(x, y, c) = Halide::saturating_cast<uint8_t>(intensity(x, y, c) / weight_sum(x, y));

    dest.compute_root().parallel(y).vectorize(x, 8).unroll(x, 4);
    return dest;
}

int main(int argc, char **argv)
{
    cv::Mat a = cv::imread("./inu.jpg", cv::IMREAD_UNCHANGED);
    cv::Mat_<cv::Vec3b> img = a.clone();
    cv::TickMeter timer;

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
    // imshow_halide("dest halide", dest_h);
    cv::imshow("halide", dest_h);
    cv::imshow("cuda", dest_cuda);
    cv::imshow("diff cuda - halide", (dest_cuda - dest_h)*50);
    cv::imshow("diff naive - halide", (dest - dest_h)*50);
    cv::waitKey();
    return 0;
}