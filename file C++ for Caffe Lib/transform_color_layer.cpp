#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/transform_color_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {    

template <typename Dtype>
void TransformColorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tTransformer Color Layer:: LayerSetUp: \t";

	if(this->layer_param_.tc_param().transform_type() == "affine") {
		transform_type_ = "affine";
	} else {
		CHECK(false) << prefix << "Transformation type only supports affine now!" << std::endl;
	}


	if(this->layer_param_.tc_param().to_compute_du()) {
		to_compute_dU_ = true;
	}

	std::cout<<prefix<<"Getting output_H_ and output_W_"<<std::endl;

	
	output_H_ = bottom[0]->shape(2);
	if(this->layer_param_.tc_param().has_output_h()) {
		output_H_ = this->layer_param_.tc_param().output_h();
	}
	output_W_ = bottom[0]->shape(3);
	if(this->layer_param_.tc_param().has_output_w()) {
		output_W_ = this->layer_param_.tc_param().output_w();
	}

	std::cout<<prefix<<"output_H_ = "<<output_H_<<", output_W_ = "<<output_W_<<std::endl;

	std::cout<<prefix<<"Getting pre-defined parameters"<<std::endl;

	is_pre_defined_theta[0] = false;
	if(this->layer_param_.tc_param().has_theta_1_1()) {
		is_pre_defined_theta[0] = true;
		++ pre_defined_count;
		pre_defined_theta[0] = this->layer_param_.tc_param().theta_1_1();
		std::cout<<prefix<<"Getting pre-defined theta[1][1] = "<<pre_defined_theta[0]<<std::endl;
	}

	is_pre_defined_theta[1] = false;
	if(this->layer_param_.tc_param().has_theta_1_2()) {
		is_pre_defined_theta[1] = true;
		++ pre_defined_count;
		pre_defined_theta[1] = this->layer_param_.tc_param().theta_1_2();
		std::cout<<prefix<<"Getting pre-defined theta[1][2] = "<<pre_defined_theta[1]<<std::endl;
	}

	is_pre_defined_theta[2] = false;
	if(this->layer_param_.tc_param().has_theta_1_3()) {
		is_pre_defined_theta[2] = true;
		++ pre_defined_count;
		pre_defined_theta[2] = this->layer_param_.tc_param().theta_1_3();
		std::cout<<prefix<<"Getting pre-defined theta[1][3] = "<<pre_defined_theta[2]<<std::endl;
	}

	is_pre_defined_theta[3] = false;
	if(this->layer_param_.tc_param().has_theta_1_4()) {
		is_pre_defined_theta[3] = true;
		++ pre_defined_count;
		pre_defined_theta[3] = this->layer_param_.tc_param().theta_1_4();
		std::cout<<prefix<<"Getting pre-defined theta[2][1] = "<<pre_defined_theta[3]<<std::endl;
	}

	is_pre_defined_theta[4] = false;
	if(this->layer_param_.tc_param().has_theta_2_1()) {
		is_pre_defined_theta[4] = true;
		++ pre_defined_count;
		pre_defined_theta[4] = this->layer_param_.tc_param().theta_2_1();
		std::cout<<prefix<<"Getting pre-defined theta[2][2] = "<<pre_defined_theta[4]<<std::endl;
	}

	is_pre_defined_theta[5] = false;
	if(this->layer_param_.tc_param().has_theta_2_2()) {
		is_pre_defined_theta[5] = true;
		++ pre_defined_count;
		pre_defined_theta[5] = this->layer_param_.tc_param().theta_2_2();
		std::cout<<prefix<<"Getting pre-defined theta[2][3] = "<<pre_defined_theta[5]<<std::endl;
	}

	
	is_pre_defined_theta[6] = false;
	if(this->layer_param_.tc_param().has_theta_2_3()) {
		is_pre_defined_theta[6] = true;
		++ pre_defined_count;
		pre_defined_theta[6] = this->layer_param_.tc_param().theta_2_3();
		std::cout<<prefix<<"Getting pre-defined theta[2][3] = "<<pre_defined_theta[6]<<std::endl;
	}
	
	
	is_pre_defined_theta[7] = false;
	if(this->layer_param_.tc_param().has_theta_2_4()) {
		is_pre_defined_theta[7] = true;
		++ pre_defined_count;
		pre_defined_theta[7] = this->layer_param_.tc_param().theta_2_4();
		std::cout<<prefix<<"Getting pre-defined theta[2][3] = "<<pre_defined_theta[7]<<std::endl;
	}
	
	is_pre_defined_theta[8] = false;
	if(this->layer_param_.tc_param().has_theta_3_1()) {
		is_pre_defined_theta[8] = true;
		++ pre_defined_count;
		pre_defined_theta[8] = this->layer_param_.tc_param().theta_3_1();
		std::cout<<prefix<<"Getting pre-defined theta[2][3] = "<<pre_defined_theta[8]<<std::endl;
	}
	
	is_pre_defined_theta[9] = false;
	if(this->layer_param_.tc_param().has_theta_3_2()) {
		is_pre_defined_theta[9] = true;
		++ pre_defined_count;
		pre_defined_theta[9] = this->layer_param_.tc_param().theta_3_2();
		std::cout<<prefix<<"Getting pre-defined theta[2][3] = "<<pre_defined_theta[9]<<std::endl;
	}
	
	is_pre_defined_theta[10] = false;
	if(this->layer_param_.tc_param().has_theta_3_3()) {
		is_pre_defined_theta[10] = true;
		++ pre_defined_count;
		pre_defined_theta[10] = this->layer_param_.tc_param().theta_3_3();
		std::cout<<prefix<<"Getting pre-defined theta[2][3] = "<<pre_defined_theta[10]<<std::endl;
	}
	
	is_pre_defined_theta[11] = false;
	if(this->layer_param_.tc_param().has_theta_3_4()) {
		is_pre_defined_theta[11] = true;
		++ pre_defined_count;
		pre_defined_theta[11] = this->layer_param_.tc_param().theta_3_4();
		std::cout<<prefix<<"Getting pre-defined theta[2][3] = "<<pre_defined_theta[11]<<std::endl;
	}
	
	
	// check the validation for the parameter theta
	//CHECK(bottom[1]->count(1) + pre_defined_count == 6) << "The dimension of theta is not six!"
	//		<< " Only " << bottom[1]->count(1) << " + " << pre_defined_count << std::endl;
	//CHECK(bottom[1]->shape(0) == bottom[0]->shape(0)) << "The first dimension of theta and " <<
	//		"U should be the same" << std::endl;

	
	// initialize the matrix for output grid
	std::cout<<prefix<<"Initializing the matrix for output grid"<<std::endl;

	vector<int> shape_output(3);
	shape_output[0] = bottom[1]->shape(0); shape_output[1] = output_H_ * output_W_; shape_output[2] = 3;
	output_grid.Reshape(shape_output);
	
	//vector<int> shape_output(2);
	//shape_output[0] = output_H_ * output_W_; shape_output[1] = 3;
	//output_grid.Reshape(shape_output);


	// scale grid cordinate (output_high and output_wide) into -1 ->1 (-1<x,y<1 )
	//Dtype* data = output_grid.mutable_cpu_data();
	//for(int i=0; i<output_H_ * output_W_; ++i) {
	//	data[3 * i] = (i / output_W_) * 1.0 / output_H_ * 2 - 1; // x axis
	//	data[3 * i + 1] = (i % output_W_) * 1.0 / output_W_ * 2 - 1; // y axis
	//	data[3 * i + 2] = 1; // z =1
	//}

	
	// initialize the matrix for input grid
	std::cout<<prefix<<"Initializing the matrix for input grid"<<std::endl;

	vector<int> shape_input(3);
	shape_input[0] = bottom[1]->shape(0); shape_input[1] = output_H_ * output_W_; shape_input[2] = 4;
	input_grid.Reshape(shape_input);

	std::cout<<prefix<<"Initialization finished."<<std::endl;
}

template <typename Dtype>
void TransformColorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tSpatial Transformer Layer:: Reshape: \t";

	if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

	N = bottom[0]->shape(0);
	C = bottom[0]->shape(1);
	H = bottom[0]->shape(2);
	W = bottom[0]->shape(3);

	// reshape V
	vector<int> shape(4);

	shape[0] = N;
	shape[1] = C;
	shape[2] = output_H_;
	shape[3] = output_W_;

	top[0]->Reshape(shape);

	// reshape dTheta_tmp
	vector<int> dTheta_tmp_shape(4);

	dTheta_tmp_shape[0] = N;
	dTheta_tmp_shape[1] = 3;
	dTheta_tmp_shape[2] = 4;
	dTheta_tmp_shape[3] = output_H_ * output_W_;

	dTheta_tmp.Reshape(dTheta_tmp_shape);

	// init all_ones_2
	vector<int> all_ones_2_shape(1);
	all_ones_2_shape[0] = output_H_ * output_W_;
	all_ones_2.Reshape(all_ones_2_shape);

	// reshape full_theta
	vector<int> full_theta_shape(2);
	full_theta_shape[0] = N;
	full_theta_shape[1] = 12;
	full_theta.Reshape(full_theta_shape);

	if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}


template <typename Dtype>
void TransformColorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tSpatial Transformer Layer:: Forward_cpu: \t";

	//CHECK(false) << "Don't use the CPU implementation! If you really want to, delete the" <<
	//		" CHECK in st_layer.cpp file. Line number: 240-241." << std::endl;

	if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

	const Dtype* U = bottom[0]->cpu_data();
	const Dtype* theta = bottom[1]->cpu_data();
	
	Dtype* input_grid_data = input_grid.mutable_cpu_data();	
	Dtype* output_grid_data = output_grid.mutable_cpu_data();
	Dtype* V = top[0]->mutable_cpu_data();

	caffe_set(input_grid.count(), (Dtype)0, input_grid_data);
	caffe_set(output_grid.count(), (Dtype)0, output_grid_data);
	caffe_set(top[0]->count(), (Dtype)0, V);

	
	// for each input
	for(int i = 0; i < N; ++i) {
		
		for (int a=0;a < (output_H_ * output_W_);a++){
			
			input_grid_data[4*i*output_H_ * output_W_+4*a] = U[i*output_H_ * output_W_*C+a];
			input_grid_data[4*i*output_H_ * output_W_+4*a+1] = U[i*output_H_ * output_W_*C + a + output_H_ * output_W_];
			input_grid_data[4*i*output_H_ * output_W_+4*a+2] = U[i*output_H_ * output_W_*C + a + 2*output_H_ * output_W_];
			input_grid_data[4*i*output_H_ * output_W_+4*a+3] = 1;
						
		}
			
		Dtype* Output_address = output_grid_data + (output_H_ * output_W_ * 3) * i;
		Dtype* Input_address = input_grid_data + (output_H_ * output_W_ * 4) * i;
		
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, output_H_ * output_W_, 3, 4, (Dtype)1.,
		      Input_address, theta + 12 * i, (Dtype)0., Output_address);
			  
		
		for (int b=0;b < (output_H_ * output_W_);b++){
			
			V[i*output_H_ * output_W_*C + b] = 	Output_address[3*b];
			V[i*output_H_ * output_W_*C + b + output_H_ * output_W_] = 	Output_address[3*b + 1];
			V[i*output_H_ * output_W_*C + b + 2*output_H_ * output_W_] = 	Output_address[3*b + 2];
					
		}
	
	}

	if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}


template <typename Dtype>
void TransformColorLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

		string prefix = "\t\tSpatial Transformer Layer:: Backward_cpu: \t";

		//CHECK(false) << "Don't use the CPU implementation! If you really want to, delete the" <<
		//		" CHECK in st_layer.cpp file. Line number: 420-421." << std::endl;

		if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

		const Dtype* dV = top[0]->cpu_diff();
		const Dtype* input_grid_data = input_grid.cpu_data();
		const Dtype* U = bottom[0]->cpu_data();
		const Dtype* theta = bottom[1]->cpu_data();

		Dtype* dU = bottom[0]->mutable_cpu_diff();
		Dtype* dTheta = bottom[1]->mutable_cpu_diff();
		//Dtype* input_grid_diff = input_grid.mutable_cpu_diff();

		caffe_set(bottom[0]->count(), (Dtype)0, dU);
		caffe_set(bottom[1]->count(), (Dtype)0, dTheta);
		//caffe_set(input_grid.count(), (Dtype)0, input_grid_diff);

		for(int i = 0; i < N; ++i) {

			for(int c=0;c < (output_H_ * output_W_);c++){
					
					// backprobagation for U
					
				//dU[i*output_H_*output_W_*C + c] += dV[i*output_H_*output_W_*C + c] * theta[12 * i] + dV[i*output_H_*output_W_*C + c + output_H_*output_W_] * theta[12 * i+4] + dV[i*output_H_*output_W_*C + c + output_H_*output_W_] * theta[12 * i+8];
				//dU[i*output_H_*output_W_*C + c + output_H_*output_W_] += dV[i*output_H_*output_W_*C + c] * theta[12 * i+1] + dV[i*output_H_*output_W_*C + c + output_H_*output_W_] * theta[12 * i+5] + dV[i*output_H_*output_W_*C + c + 2*output_H_*output_W_] * theta[12 * i+9];
				//dU[i*output_H_*output_W_*C + c + 2*output_H_*output_W_] += dV[i*output_H_*output_W_*C + c] * theta[12 * i+2] + dV[i*output_H_*output_W_*C + c + output_H_*output_W_] * theta[12 * i+6] + dV[i*output_H_*output_W_*C + c + 2*output_H_*output_W_] * theta[12 * i+10];
					
					// backprobagation for theta
				dTheta[12 * i]     +=  dV[i*output_H_*output_W_*C + c]*U[i*output_H_*output_W_*C + c];
				dTheta[12 * i + 1] +=  dV[i*output_H_*output_W_*C + c]*U[i*output_H_*output_W_*C + c + output_H_*output_W_];
				dTheta[12 * i + 2] +=  dV[i*output_H_*output_W_*C + c]*U[i*output_H_*output_W_*C + c + 2*output_H_*output_W_];
				dTheta[12 * i + 3] +=  dV[i*output_H_*output_W_*C + c];
					
					
				dTheta[12 * i + 4] +=  dV[i*output_H_*output_W_*C + c + output_H_*output_W_]*U[i*output_H_*output_W_*C + c];
				dTheta[12 * i + 5] +=  dV[i*output_H_*output_W_*C + c + output_H_*output_W_]*U[i*output_H_*output_W_*C + c + output_H_*output_W_]; 
				dTheta[12 * i + 6] +=  dV[i*output_H_*output_W_*C + c + output_H_*output_W_]*U[i*output_H_*output_W_*C + c + 2*output_H_*output_W_];
				dTheta[12 * i + 7] +=  dV[i*output_H_*output_W_*C + c + output_H_*output_W_];
					
				dTheta[12 * i + 8] +=  dV[i*output_H_*output_W_*C + c + 2*output_H_*output_W_]*U[i*output_H_*output_W_*C + c];
				dTheta[12 * i + 9] +=  dV[i*output_H_*output_W_*C + c + 2*output_H_*output_W_]*U[i*output_H_*output_W_*C + c + output_H_*output_W_]; 
				dTheta[12 * i + 10] += dV[i*output_H_*output_W_*C + c + 2*output_H_*output_W_]*U[i*output_H_*output_W_*C + c + 2*output_H_*output_W_];
				dTheta[12 * i + 11] += dV[i*output_H_*output_W_*C + c + 2*output_H_*output_W_];
																				
			}
				
		}

		if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;

}
#ifdef CPU_ONLY
STUB_GPU(TransformColorLayer);
#endif

INSTANTIATE_CLASS(TransformColorLayer);
REGISTER_LAYER_CLASS(TransformColor);

}  // namespace caffe
