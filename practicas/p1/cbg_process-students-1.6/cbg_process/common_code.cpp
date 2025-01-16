#include "common_code.hpp"

cv::Mat
fsiv_convert_image_byte_to_float(const cv::Mat &img)
{
    CV_Assert(img.depth() == CV_8U);
    cv::Mat out;
    //! TODO
    // Hint: use cv::Mat::convertTo().
    float scale = 1/255.0f;
    img.convertTo(out, CV_32F,scale);
    //
    CV_Assert(out.rows == img.rows && out.cols == img.cols);
    CV_Assert(out.depth() == CV_32F);
    CV_Assert(img.channels() == out.channels());
    return out;
}

cv::Mat
fsiv_convert_image_float_to_byte(const cv::Mat &img)
{
    CV_Assert(img.depth() == CV_32F);
    cv::Mat out;
    //! TODO
    // Hint: use cv::Mat::convertTo()
    int scale = 255;
    img.convertTo(out, CV_8U, scale);
    //
    CV_Assert(out.rows == img.rows && out.cols == img.cols);
    CV_Assert(out.depth() == CV_8U);
    CV_Assert(img.channels() == out.channels());
    return out;
}

cv::Mat
fsiv_convert_bgr_to_hsv(const cv::Mat &img)
{
    CV_Assert(img.channels() == 3);
    cv::Mat out;
    //! TODO
    // Hint: use cvtColor.
    // Remember: the input color scheme is assumed to be BGR.
    cv::cvtColor(img, out, cv::COLOR_BGR2HSV);
    //
    CV_Assert(out.channels() == 3);
    return out;
}

cv::Mat
fsiv_convert_hsv_to_bgr(const cv::Mat &img)
{
    CV_Assert(img.channels() == 3);
    cv::Mat out;
    //! TODO
    // Hint: use cvtColor.
    // Remember: the ouput color scheme is assumed to be BGR.
    cv::cvtColor(img, out, cv::COLOR_HSV2BGR);
    //
    CV_Assert(out.channels() == 3);
    return out;
}

cv::Mat
fsiv_cbg_process(const cv::Mat &in,
                 double contrast, double brightness, double gamma,
                 bool only_luma)
{
    CV_Assert(in.depth() == CV_8U);
    cv::Mat out;
    // TODO
    // Hint: convert to float range [0,1] before processing the image.
    // Hint: use cv::pow() to apply the gamma parameter.
    // Hint: if input channels is 3 and only luma is required, convert to HSV
    //       color space and process only de V (luma) channel.
    bool is_bgr = in.channels() == 3;
    cv::Mat float_in = fsiv_convert_image_byte_to_float(in);

    if (is_bgr && only_luma)
    {
        cv::Mat hsv = fsiv_convert_bgr_to_hsv(float_in);
        std::vector<cv::Mat> channels;

        cv::split(hsv, channels);
        cv::Mat &luma = channels.at(2);

        cv::pow(luma, gamma, luma);
        luma *= contrast;
        luma += brightness;

        cv::merge(channels, out);

        out = fsiv_convert_hsv_to_bgr(out);
    }
    else
    {
        cv::pow(float_in, gamma, out);
        cv::multiply(cv::Scalar::all(contrast), out, out);
        out += cv::Scalar::all(brightness);

    }

    out = fsiv_convert_image_float_to_byte(out);
    //
    CV_Assert(out.rows == in.rows && out.cols == in.cols);
    CV_Assert(out.depth() == CV_8U);
    CV_Assert(out.channels() == in.channels());
    return out;
}
