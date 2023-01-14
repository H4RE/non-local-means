#pragma once

void non_local_means_CUDA(cv::Mat& src, cv::Mat& dest, int search_window, int patch_radius, float stdev_noise, float filter_param_h);