#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "common_code.hpp"

std::vector<cv::Point3f>
fsiv_generate_3d_calibration_points(const cv::Size &board_size,
                                    float square_size)
{
    std::vector<cv::Point3f> ret_v;
    // TODO
    // Remember: the first inner point has (1,1) in board coordinates.
    const size_t width = board_size.width;
    const size_t height = board_size.height;
    const size_t total = width * height;

    ret_v.reserve(total);

    for (size_t i = 1; i <= height; i++)
        for (size_t j = 1; j <= width; j++)
            ret_v.emplace_back(cv::Point3d(j, i, 0) * square_size);

    //
    CV_Assert(ret_v.size() ==
              static_cast<size_t>(board_size.width * board_size.height));
    return ret_v;
}

bool fsiv_fast_find_chessboard_corners(const cv::Mat &img, const cv::Size &board_size,
                                       std::vector<cv::Point2f> &corner_points)
{
    CV_Assert(img.type() == CV_8UC3);
    bool was_found = false;
    // TODO
    // Hint: use cv::findChessboardCorners adding fast check to the defaults flags.
    // Remember: do not refine the corner points to get a better computational performance.
    was_found = cv::findChessboardCorners(img, board_size, corner_points, 8);
    //
    return was_found;
}

void fsiv_compute_camera_pose(const std::vector<cv::Point3f> &_3dpoints,
                              const std::vector<cv::Point2f> &_2dpoints,
                              const cv::Mat &camera_matrix,
                              const cv::Mat &dist_coeffs,
                              cv::Mat &rvec,
                              cv::Mat &tvec)
{
    CV_Assert(_3dpoints.size() >= 4 && _3dpoints.size() == _2dpoints.size());
    // TODO
    // Hint: use cv::solvePnP to the pose of a calibrated camera.
    cv::solvePnP(_3dpoints, _2dpoints, camera_matrix, dist_coeffs, rvec, tvec);
    //
    CV_Assert(rvec.rows == 3 && rvec.cols == 1 && rvec.type() == CV_64FC1);
    CV_Assert(tvec.rows == 3 && tvec.cols == 1 && tvec.type() == CV_64FC1);
}

void fsiv_draw_axes(cv::Mat &img,
                    const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
                    const cv::Mat &rvec, const cv::Mat &tvec,
                    const float size, const int line_width)
{
    // TODO
    // Hint: use cv::projectPoints to get the image coordinates of the 3D points
    // (0,0,0), (size, 0, 0), (0, size, 0) and (0, 0, -size) and draw a line for
    // each axis: blue for axis OX, green for axis OY and red for axis OZ.
    // Warning: use of cv::drawFrameAxes() is not allowed.

    // Recuerda BGR!!!

    std::vector<cv::Point3f> axes_points = 
    {
        { 0,   0,   0},  //0
        {size, 0,   0},  //x
        { 0,  size, 0},  //y
        { 0,   0, -size} //z
    };

    std::vector<cv::Point2f> image_points;

    cv::projectPoints(axes_points, rvec, tvec, camera_matrix, dist_coeffs, image_points);

    cv::line(img, image_points.at(0), image_points.at(1), cv::Scalar(255, 0, 0), line_width);
    cv::line(img, image_points.at(0), image_points.at(2), cv::Scalar(0, 255, 0), line_width);
    cv::line(img, image_points.at(0), image_points.at(3), cv::Scalar(0, 0, 255), line_width);
    //
}

void fsiv_load_calibration_parameters(cv::FileStorage &fs,
                                      cv::Size &camera_size,
                                      float &error,
                                      cv::Mat &camera_matrix,
                                      cv::Mat &dist_coeffs,
                                      cv::Mat &rvec,
                                      cv::Mat &tvec)
{
    CV_Assert(fs.isOpened());
    // TODO
    //  Hint: use fs["label"] >> var to load data items from the file.
    //  @see cv::FileStorage operators "[]" and ">>"
    fs["image-width"] >> camera_size.width;
    fs["image-height"] >> camera_size.height;
    fs["error"] >> error;
    fs["camera-matrix"] >> camera_matrix;
    fs["distortion-coefficients"] >> dist_coeffs;
    fs["rvec"] >> rvec;
    fs["tvec"] >> tvec;
    //
    CV_Assert(fs.isOpened());
    CV_Assert(camera_matrix.type() == CV_64FC1 && camera_matrix.rows == 3 && camera_matrix.cols == 3);
    CV_Assert(dist_coeffs.type() == CV_64FC1 && dist_coeffs.rows == 1 && dist_coeffs.cols == 5);
    CV_Assert(rvec.type() == CV_64FC1 && rvec.rows == 3 && rvec.cols == 1);
    CV_Assert(tvec.type() == CV_64FC1 && tvec.rows == 3 && tvec.cols == 1);
    return;
}

void fsiv_draw_3d_model(cv::Mat &img, const cv::Mat &M, const cv::Mat &dist_coeffs,
                        const cv::Mat &rvec, const cv::Mat &tvec,
                        const float size)
{
    CV_Assert(img.type() == CV_8UC3);

    // TODO
    // Hint: build a 3D object points vector with pair of segments end points.
    // Use cv::projectPoints to get the 2D image coordinates of 3D object points,
    // build a vector of vectors of Points, one for each segment, and use
    // cv::polylines to draw the wire frame projected model.
    // Hint: use a "reference point" to move the model around the image and update it
    //       at each call to move the 3D model around the scene.

    //
}

void fsiv_project_image(const cv::Mat &model, cv::Mat &scene,
                        const cv::Size &board_size,
                        const std::vector<cv::Point2f> &chess_board_corners)
{
    CV_Assert(!model.empty() && model.type() == CV_8UC3);
    CV_Assert(!scene.empty() && scene.type() == CV_8UC3);
    CV_Assert(static_cast<size_t>(board_size.area()) ==
              chess_board_corners.size());

    // TODO
    // Hint: get the upper-left, upper-right, bottom-right and bottom-left
    //   chess_board_corners and map to the upper-left, upper-right, bottom-right
    //   and bottom-left model image point coordinates.
    //   Use cv::getPerspectiveTransform compute such mapping.
    // Hint: use cv::wrapPerspective to get a wrap version of the model image
    //   using the computed mapping. Use INTER_LINEAR as interpolation method
    //   and use BORDER_TRANSPARENT as a border extrapolation method
    //   to maintain the underlying image.
    //                    

    //No importa le orden de los puntos, la correspondencia SI                  
    const auto &chess = chess_board_corners; 

    std::vector<cv::Point2f> scene_points = {
        chess.at(0), // Top-Left
        chess.at(board_size.width - 1), // Top-Right,
        chess.at(chess.size() - board_size.width), // Bottom-Left
        chess.back() // Bottom-Right
    };

    std::vector<cv::Point2f> model_points = {
        cv::Point2f(0, 0),
        cv::Point2f(model.cols - 1, 0),
        cv::Point2f(0, model.rows - 1),
        cv::Point2f(model.cols - 1, model.rows - 1)
    };

    cv::Mat homograph = cv::getPerspectiveTransform(model_points, scene_points);
    cv::warpPerspective(model, scene, homograph, scene.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
    //
}
