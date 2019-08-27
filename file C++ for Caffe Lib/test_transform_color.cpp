#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/transform_color_layer.hpp"
#include "caffe/filler.hpp"
//#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "fstream"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class TransformColorLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  TransformColorLayerTest()
 	 : blob_U_(new Blob<Dtype>(2, 3, 10, 10)),
 	   blob_theta_(new Blob<Dtype>(2, 3, 4, 1)),
 	   blob_V_(new Blob<Dtype>(2, 3, 10, 10)) {

	  FillerParameter filler_param;
	  GaussianFiller<Dtype> filler(filler_param);
	  filler.Fill(this->blob_U_);
	  filler.Fill(this->blob_theta_);

	  vector<int> shape_theta(2);
	  shape_theta[0] = 2; shape_theta[1] = 12;
	  blob_theta_->Reshape(shape_theta);

	  blob_bottom_vec_.push_back(blob_U_);
	  blob_bottom_vec_.push_back(blob_theta_);
	  blob_top_vec_.push_back(blob_V_);
  }
  virtual ~TransformColorLayerTest() { delete blob_V_; delete blob_theta_; delete blob_U_; }
  Blob<Dtype>* blob_U_;
  Blob<Dtype>* blob_theta_;
  Blob<Dtype>* blob_V_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TransformColorLayerTest, TestDtypesAndDevices);

TYPED_TEST(TransformColorLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {

	  	// reshape theta to have full 6 dimension
	  	vector<int> shape_theta(2);
		shape_theta[0] = 2; shape_theta[1] = 12;
		this->blob_theta_->Reshape(shape_theta);

		// fill random variables for theta
		FillerParameter filler_param;
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->blob_theta_);

		// set layer_param
		LayerParameter layer_param;
		TransformColorLayer *tc_param = layer_param.mutable_tc_param();
		tc_param->set_output_h(10);
		tc_param->set_output_w(10);

		// begin to check
		TransformColorLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-6, 1e-6);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(TransformColorLayerTest, TestGradientWithPreDefinedTheta) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {

		// reshape theta to have only 2 dimensions
		vector<int> shape_theta(2);
		shape_theta[0] = 2; shape_theta[1] = 2;
		this->blob_theta_->Reshape(shape_theta);

		// fill random variables for theta
		FillerParameter filler_param;
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->blob_theta_);

		// set layer_param
		LayerParameter layer_param;
		TransformColorLayer *tc_param = layer_param.mutable_tc_param();
		tc_param->set_output_h(10);
		tc_param->set_output_w(10);

		tc_param->set_theta_1_1(0.5);
		tc_param->set_theta_1_2(0);
		tc_param->set_theta_2_1(0);
		tc_param->set_theta_2_2(0.5);

		// begin to check
		TransformColorLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-6, 1e-6);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
