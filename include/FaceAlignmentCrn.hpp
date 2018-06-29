/** ****************************************************************************
 *  @file    FaceAlignmentCrn.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2017/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_ALIGNMENT_CRN_HPP
#define FACE_ALIGNMENT_CRN_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <FaceAlignment.hpp>
#include <opencv2/opencv.hpp>
#include "tensorflow/core/public/session.h"

namespace upm {

/** ****************************************************************************
 * @class FaceAlignmentCrn
 * @brief Class used for facial feature point detection.
 ******************************************************************************/
class FaceAlignmentCrn: public FaceAlignment
{
public:
  FaceAlignmentCrn(std::string path) : _path(path) {};

  ~FaceAlignmentCrn() {};

  void
  parseOptions
    (
    int argc,
    char **argv
    );

  void
  train
    (
    const std::vector<FaceAnnotation> &anns_train,
    const std::vector<FaceAnnotation> &anns_valid
    );

  void
  load();

  tensorflow::Status
  imageToTensor
    (
    const cv::Mat &img,
    std::vector<tensorflow::Tensor>* output_tensors
    );

  std::vector<cv::Mat>
  tensorToMaps
    (
    const tensorflow::Tensor &img_tensor,
    const cv::Size &face_size
    );

  void
  process
    (
    cv::Mat frame,
    std::vector<FaceAnnotation> &faces,
    const FaceAnnotation &ann
    );

private:
  std::string _path;
  std::vector<unsigned int> _cnn_landmarks;
  std::unique_ptr<tensorflow::Session> _session;
};

} // namespace upm

#endif /* FACE_ALIGNMENT_CRN_HPP */
