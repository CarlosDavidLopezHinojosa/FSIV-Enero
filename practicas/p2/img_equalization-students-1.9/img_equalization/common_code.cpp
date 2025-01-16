#include <iostream>
#include "common_code.hpp"
#include <opencv2/imgproc.hpp>

cv::Mat
fsiv_compute_image_histogram(const cv::Mat &in)
{
    CV_Assert(in.type() == CV_8UC1);
    cv::Mat hist;
    // TODO
    // Hint: Use the function cv::calcHist.
    std::vector<cv::Mat> images = {in};
    std::vector<int> channels = {0};
    std::vector<int> hist_size = {256};
    std::vector<float> ranges = {0, 256};

    cv::calcHist(images, channels, cv::noArray(), hist, hist_size, ranges);

    //
    CV_Assert(!hist.empty());
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.rows == 256 && hist.cols == 1);
    return hist;
}

void fsiv_normalize_histogram(cv::Mat &hist)
{
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.rows == 256 && hist.cols == 1);

    // TODO
    // Hint: Use the function cv::normalize() with norm L1 = 1.0
    cv::normalize(hist, hist, 1.0f, 0.0f, cv::NORM_L1);
    //

    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.rows == 256 && hist.cols == 1);
    CV_Assert(cv::sum(hist)[0] == 0.0 || cv::abs(cv::sum(hist)[0] - 1.0) <= 1.0e-6);
}

void fsiv_accumulate_histogram(cv::Mat &hist)
{
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.rows == 256 && hist.cols == 1);

    // TODO
    for (int i = 1; i < hist.rows; i++)
        hist.at<float>(i) += hist.at<float>(i - 1);
    //

    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.rows == 256 && hist.cols == 1);
}

void fsiv_compute_clipped_histogram(cv::Mat &h, float cl)
{
    CV_Assert(h.type() == CV_32FC1);
    CV_Assert(h.rows == 256 && h.cols == 1);

    // TODO
    // Remember: the ouput histogram will be clipped to cl value and the
    // residual area will be redistributed equally in all the histogram's bins.
    float residual_area = 0.0f;
    for (int i = 0; i < h.rows; i++)
    {
        if (h.at<float>(i) > cl)
        {
            residual_area += h.at<float>(i) - cl;
            h.at<float>(i) = cl;
        }
    }

    h += (residual_area / h.rows);
    //

    CV_Assert(h.type() == CV_32FC1);
    CV_Assert(h.rows == 256 && h.cols == 1);
}

float fsiv_compute_actual_clipping_histogram_value(const cv::Mat &h, float s)
{
    CV_Assert(h.type() == CV_32FC1);
    CV_Assert(h.rows == 256 && h.cols == 1);

    int CL = s * cv::sum(h)[0] / h.rows;

    // TODO: Code the algorithm shown in the practical assignment description.
    int bottom = 0;
    int top = CL;
    int n = int(h.rows);

    while (top - bottom > 1)
    {
        int middle = (top + bottom) / 2;
        int residual = 0;

        for (int i = 0; i < n; i++)
            if(h.at<float>(i) > middle)
                residual += h.at<float>(i) - middle;
        
        if (residual > (CL - middle) * n)
            top = middle;
        else
            bottom = middle;
    }
    
    CL = (top + bottom) / 2;

    //

    return CL;
}

cv::Mat
fsiv_create_equalization_lookup_table(const cv::Mat &hist,
                                      float s)
{
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.rows == 256 && hist.cols == 1);
    cv::Mat lkt = hist.clone();

    if (s >= 1.0)
    {
        // TODO: Clip the histogram.
        // Hint: use fsiv_compute_actual_clipping_histogram_value to compute the
        //       clipping level.
        // Hint: use fsiv_compute_clipped_histogram to clip the histogram.
        float cl = fsiv_compute_actual_clipping_histogram_value(lkt, s);
        fsiv_compute_clipped_histogram(lkt, cl);
        //
    }

    // TODO: Build the equalization transform function.
    // Remember: the transform function will be the accumulated normalized
    //           image histogram.
    // Hint: use cv::Mat::convertTo() method to convert the float range [0.0, 1.0]
    //       to [0, 255] byte range.
    //
    fsiv_normalize_histogram(lkt);
    fsiv_accumulate_histogram(lkt);
    lkt.convertTo(lkt, CV_8UC1, 255);
    //

    CV_Assert(lkt.type() == CV_8UC1);
    CV_Assert(lkt.rows == 256 && lkt.cols == 1);
    return lkt;
}

cv::Mat
fsiv_apply_lookup_table(const cv::Mat &in, const cv::Mat &lkt,
                        cv::Mat &out)
{
    CV_Assert(in.type() == CV_8UC1);
    CV_Assert(lkt.type() == CV_8UC1);
    CV_Assert(lkt.rows == 256 && lkt.cols == 1);
    CV_Assert(out.empty() || (out.type() == CV_8UC1 &&
                              out.rows == in.rows && out.cols == in.cols));

    // TODO
    // Hint: you can use the cv::LUT function.
    cv::LUT(in, lkt, out);
    //
    CV_Assert(out.rows == in.rows && out.cols == in.cols && out.type() == in.type());
    return out;
}
