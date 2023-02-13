#include <Halide.h>

class NonLocalMeans : public Halide::Generator<NonLocalMeans>
{
public:
    Input<Halide::Buffer<uint8_t, 3>> src{"src"};
    Input<int> search_radius{"search_radius"};
    Input<int> patch_radius{"patch_radius"};
    Input<float> stdev_noise{"stdev_noise"};
    Input<float> filter_param_h{"filter_param_h"};

    Output<Halide::Buffer<uint8_t, 3>> dest{"dest"};
    void generate()
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

        const Halide::Expr stdev_noise2 = stdev_noise * stdev_noise;
        const Halide::Expr filter_param_h2 = filter_param_h * filter_param_h;
        const Halide::Expr patch_sizeXch = (patch_radius * 2 + 1) * (patch_radius * 2 + 1) * 3;

        weight(x, y, dx, dy) = Halide::exp(-Halide::max(norm_sum(x, y, dx, dy) / patch_sizeXch - 2 * stdev_noise2, 0) / filter_param_h2);
        Halide::RDom swin(-search_radius, search_radius * 2 + 1, -search_radius, search_radius * 2 + 1);
        Halide::Func weight_sum;
        weight_sum(x, y) += weight(x, y, swin.x, swin.y);
        Halide::Func intensity;
        intensity(x, y, c) += weight(x, y, swin.x, swin.y) * srcf_ex(x + swin.x, y + swin.y, c);

        // Halide::Func dest;
        dest(x, y, c) = Halide::saturating_cast<uint8_t>(intensity(x, y, c) / weight_sum(x, y));

        dest.compute_root().vectorize(x, 16).parallel(y);
    }
};

HALIDE_REGISTER_GENERATOR(NonLocalMeans, non_local_means_halide)