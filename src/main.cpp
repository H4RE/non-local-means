#include "Halide.h"
#include "halide_image_io.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <type_traits>
#include "non_local_means.hpp"
inline __m512 _mm512_cvt_uchar2float(__m128i xmm)
{
    return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(xmm));
}
inline __m128i _mm512_cvt_float2uchar(__m512 zmm)
{
    return _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(zmm));
}
void print_m512(__m512 a)
{
    for (int i = 0; i < 16; ++i)
    {
        std::cout << a.m512_f32[i] << std::endl;
    }
}
inline void mm_load_and_split_bgr(const uchar *src_ptr, __m512 &BBB, __m512 &GGG, __m512 &RRR)
{
    const int mask1 = 0b0000'0000'0001'1111;
    const int mask2 = 0b0000'0011'1110'0000;
    // load
    __m128i msrc_1 = _mm_loadu_epi8(src_ptr);
    __m128i msrc_2 = _mm_loadu_epi8(src_ptr + 16);
    __m128i msrc_3 = _mm_loadu_epi8(src_ptr + 32);

    // uchar -> int -> float
    __m512 src_f1 = _mm512_cvt_uchar2float(msrc_1);
    __m512 src_f2 = _mm512_cvt_uchar2float(msrc_2);
    __m512 src_f3 = _mm512_cvt_uchar2float(msrc_3);
    // f1 BGRB GRBG | RBGR BGRB
    // -> BBBB BBGG | GGGR RRRR
    const __m512i idx = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14);
    __m512 a = _mm512_permutexvar_ps(idx, src_f1);

    // f2 GRBG RBGR | BGRB GRBG
    // -> GGGG GGRR | RRRB BBBB
    __m512 b = _mm512_permutexvar_ps(idx, src_f2);

    // f3 RBGR BGRB | GRBG RBGR
    // -> RRRR RRBB | BBBG GGGG
    __m512 c = _mm512_permutexvar_ps(idx, src_f3);

    // blend(blend(a,b),c)-->BBBB
    __m512 temp = _mm512_mask_blend_ps(mask1, a, b);
    BBB = _mm512_mask_blend_ps(mask2, temp, c);
    // blend(blend(b,c), a)-->GGGG
    temp = _mm512_mask_blend_ps(mask1, b, c);
    GGG = _mm512_mask_blend_ps(mask2, temp, a);
    // blend(blend(a,c),b) -->RRRR
    temp = _mm512_mask_blend_ps(mask1, c, a);
    RRR = _mm512_mask_blend_ps(mask2, temp, b);
    return;
}

void non_local_means_AVX512(cv::Mat_<cv::Vec3b> &src_, cv::Mat_<cv::Vec3b> &dest, int search_radius, int patch_radius, float stdev_noise, float filter_param_h)
{
    const int ch = 3;
    const int offset = search_radius + patch_radius;
    const int remain = (src_.cols * ch) % 48;
    int pad = remain != 0 ? 48 - remain : 0;
    pad += pad % ch;

    const int end = (src_.cols - offset) * ch - remain;
    const __m512 patchsizexch_inv = _mm512_div_ps(_mm512_set1_ps(1.f), _mm512_set1_ps(ch * (patch_radius * 2 + 1) * (patch_radius * 2 + 1)));
    const __m512 stdev_noise2 = _mm512_set1_ps(stdev_noise * stdev_noise);
    __m512 filter_param_h2neginv = _mm512_set1_ps(filter_param_h * filter_param_h);
    filter_param_h2neginv = _mm512_sub_ps(_mm512_setzero_ps(), _mm512_div_ps(_mm512_set1_ps(1.f), filter_param_h2neginv));

    cv::Mat src;
    cv::copyMakeBorder(src_, src, offset, offset, offset, offset + (pad / 3), cv::BORDER_REFLECT101);

#pragma omp parallel for
    for (int y = offset; y < src.rows - offset; y++)
    {
        uchar *dest_ptr = dest.data + (y - offset) * dest.cols * ch;
        for (int x = offset * ch; x < src.cols * ch - offset * ch - pad; x += 48)
        {
            __m512 sum = _mm512_setzero_ps();
            __m512 intensity_b = _mm512_setzero_ps();
            __m512 intensity_g = _mm512_setzero_ps();
            __m512 intensity_r = _mm512_setzero_ps();
            const int mask1 = 0b0000'0000'0001'1111;
            const int mask2 = 0b0000'0011'1110'0000;
            for (int sy = -search_radius; sy <= search_radius; sy++)
            {
                for (int sx = -search_radius; sx <= search_radius; sx++)
                {
                    __m512 norm = _mm512_setzero_ps();
                    for (int py = -patch_radius; py <= patch_radius; py++)
                    {
                        for (int px = -patch_radius; px <= patch_radius; px++)
                        {
                            uchar const *src_ptr = src.data + src.cols * ch * (y + py) + x + px * ch;
                            uchar const *src_ptr1 = src.data + src.cols * ch * (y + py + sy) + x + sx * ch + px * ch;
                            __m512 BBB, GGG, RRR, BBB_, GGG_, RRR_;
                            mm_load_and_split_bgr(src_ptr, BBB, GGG, RRR);
                            mm_load_and_split_bgr(src_ptr1, BBB_, GGG_, RRR_);

                            __m512 sub = _mm512_sub_ps(BBB, BBB_);
                            norm = _mm512_fmadd_ps(sub, sub, norm);
                            sub = _mm512_sub_ps(GGG, GGG_);
                            norm = _mm512_fmadd_ps(sub, sub, norm);
                            sub = _mm512_sub_ps(RRR, RRR_);
                            norm = _mm512_fmadd_ps(sub, sub, norm);
                        }
                    } // py
                    // load
                    uchar const *src_ptr2 = src.data + src.cols * ch * (y + sy) + x + sx * ch;
                    __m512 BBB, GGG, RRR;
                    mm_load_and_split_bgr(src_ptr2, BBB, GGG, RRR);

                    __m512 weight = _mm512_fmsub_ps(norm, patchsizexch_inv, _mm512_mul_ps(_mm512_set1_ps(2.f), stdev_noise2));
                    weight = _mm512_max_ps(weight, _mm512_setzero_ps());
                    weight = _mm512_mul_ps(weight, filter_param_h2neginv);
                    weight = _mm512_exp_ps(weight);
                    sum = _mm512_add_ps(weight, sum);
                    intensity_b = _mm512_fmadd_ps(weight, BBB, intensity_b);
                    intensity_g = _mm512_fmadd_ps(weight, GGG, intensity_g);
                    intensity_r = _mm512_fmadd_ps(weight, RRR, intensity_r);

                } // sx
            }     // sy
            /// TODO: replace saturate_cast
            intensity_b = _mm512_div_round_ps(intensity_b, sum, _MM_ROUND_MODE_NEAREST);
            intensity_g = _mm512_div_round_ps(intensity_g, sum, _MM_ROUND_MODE_NEAREST);
            intensity_r = _mm512_div_round_ps(intensity_r, sum, _MM_ROUND_MODE_NEAREST);

            // -> BBBB BBGG | GGGB BBBB -> BBBB BBGG | GGGR RRRR
            __m512 dest_1 = _mm512_mask_blend_ps(mask2, intensity_b, intensity_g);
            dest_1 = _mm512_mask_blend_ps(mask1, dest_1, intensity_r);
            // BBBB BBGG | GGGR RRRR -> BGRB GRBG | RBGR BGRB
            __m512i idx_2 = _mm512_setr_epi32(0, 6, 11, 1, 7, 12, 2, 8, 13, 3, 9, 14, 4, 10, 15, 5);
            dest_1 = _mm512_permutexvar_ps(idx_2, dest_1);

            // - >GGGG GGRR | RRRG GGGG -> GGGG GGRR | RRRB BBBB
            __m512 dest_2 = _mm512_mask_blend_ps(mask2, intensity_g, intensity_r);
            dest_2 = _mm512_mask_blend_ps(mask1, dest_2, intensity_b);
            dest_2 = _mm512_permutexvar_ps(idx_2, dest_2);
            // - >RRR RRBB | BBBR RRRR -> RRR RRBB | BBBG GGGG
            __m512 dest_3 = _mm512_mask_blend_ps(mask2, intensity_r, intensity_b);
            dest_3 = _mm512_mask_blend_ps(mask1, dest_3, intensity_g);
            dest_3 = _mm512_permutexvar_ps(idx_2, dest_3);

            // float -> uchar
            __m128i dest_u1 = _mm512_cvt_float2uchar(dest_1);
            __m128i dest_u2 = _mm512_cvt_float2uchar(dest_2);
            __m128i dest_u3 = _mm512_cvt_float2uchar(dest_3);

            const int simd_part = src.cols * ch - remain - pad - offset * ch;
            if (x < simd_part)
            {
                _mm_storeu_epi8(dest_ptr, dest_u1);
                _mm_storeu_epi8(dest_ptr + 16, dest_u2);
                _mm_storeu_epi8(dest_ptr + 32, dest_u3);
            }
            else
            {
                for (int i = 0; i < remain; i++)
                {
                    if (i < 16)
                    {
                        dest_ptr[i] = dest_u1.m128i_u8[i];
                    }
                    else if (i < 32)
                    {
                        dest_ptr[i] = dest_u2.m128i_u8[i];
                    }
                    else
                    {
                        dest_ptr[i] = dest_u3.m128i_u8[i];
                    }
                }
            }
            dest_ptr += 48;
        } // x
    }     // y
}

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

    weight(x, y, dx, dy) = Halide::exp(-Halide::max(norm_sum(x, y, dx, dy) / (float)patch_sizeXch - 2 * stdev_noise2, 0) / filter_param_h2);
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
    // cv::Mat a = cv::Mat::ones(cv::Size(512,128), CV_8UC3);
    // a*=255;
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
    // cv::imshow("AVX", dest_AVX);
    // cv::imshow("in", a);
    // cv::imshow("diff", (a-dest_AVX)*100);
    // cv::imwrite("./inu_.jpg", dest_AVX);
    // cv::waitKey();
    // return 0;

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
    // imshow_halide("dest halide", dest_h);
    cv::imshow("halide", dest_h);
    cv::imshow("cuda", dest_cuda);
    cv::imshow("diff cuda - halide", (dest_cuda - dest_h) * 5);
    cv::imshow("diff naive - halide", (dest - dest_h) * 5);
    cv::imshow("diff AVX - halide", (dest_AVX - dest_h) * 5);
    cv::waitKey();
    return 0;
}