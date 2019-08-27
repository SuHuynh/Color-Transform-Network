#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/layers/transform_color_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
__global__ void set_value_to_constant(const int nthreads, Dtype value, int size, 
	int i, Dtype* dst) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		dst[index * size + i] = value;
	}
}


template <typename Dtype>
__global__ void copy_values(const int nthreads, int size_src, int k, 
	const Dtype* src, int size_dst, int i, Dtype* dst) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		dst[index * size_dst + i] = src[index * size_src + k];
	}
}


//////////////////////// added by SU HUYNH VAN//////////

template <typename Dtype>
__global__ void Compute_input_fromU(const int nthreads, const Dtype* UU, Dtype* input_U, int output_H_, int output_W_, int C) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int i = index / (output_W_ * output_H_);

		int idx = s * output_W_ + t;

		input_U[4 * i*output_W_ * output_H_ + 4 * idx] = UU[i*output_W_ * output_H_*C + idx];
		input_U[4 * i*output_W_ * output_H_ + 4 * idx + 1] = UU[i*output_W_ * output_H_*C + idx + output_W_ * output_H_];
		input_U[4 * i*output_W_ * output_H_ + 4 * idx + 2] = UU[i*output_W_ * output_H_*C + idx + 2 * output_W_ * output_H_];
		input_U[4 * i*output_W_ * output_H_ + 4 * idx + 3] = 1;
			
	}
}

template <typename Dtype>
__global__ void ComputeV_from_output(const int nthreads, Dtype* VV, Dtype* ouput_V, int output_H_, int output_W_, int C) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int i = index / (output_W_ * output_H_);

		int idx = s * output_W_ + t;
		
		VV[i*output_W_ * output_H_*C + idx] = ouput_V[3 * i*output_W_ * output_H_ + 3 * idx];
		VV[i*output_W_ * output_H_*C + idx + output_W_ * output_H_] = ouput_V[3 * i*output_W_ * output_H_ + 3 * idx + 1];
		VV[i*output_W_ * output_H_*C + idx + 2 * output_W_ * output_H_] = ouput_V[3 * i*output_W_ * output_H_ + 3 * idx + 2];
		
	}
}

/////////////////////END/////////////////////



template <typename Dtype>
void TransformColorLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	string prefix = "TransformColorLayer::Forward_gpu::\t";

	const Dtype* U = bottom[0]->gpu_data();
	const Dtype* theta = bottom[1]->gpu_data();
	
	Dtype* full_theta_data = full_theta.mutable_gpu_data();
	Dtype* input_grid_data = input_grid.mutable_gpu_data();	
	Dtype* output_grid_data = output_grid.mutable_gpu_data();
	Dtype* V = top[0]->mutable_gpu_data();

	caffe_gpu_set(input_grid.count(), (Dtype)0, input_grid_data);
	caffe_gpu_set(output_grid.count(), (Dtype)0, output_grid_data);
	caffe_gpu_set(top[0]->count(), (Dtype)0, V);
	
	// compute full_theta
	int k = 0; 
	const int num_threads = N;
	for(int i=0; i<12; ++i) {
		if(is_pre_defined_theta[i]) {
			set_value_to_constant<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>( 
				num_threads, pre_defined_theta[i], 12, i, full_theta_data);
			//std::cout << "Setting value " << pre_defined_theta[i] << " to "<< i << 
			//	"/6 of full_theta_data" << std::endl;
		} else {
			copy_values<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(num_threads, 
				12 - pre_defined_count, k, theta, 12, i, full_theta_data);
			//std::cout << "Copying " << k << "/" << 6 - pre_defined_count << " of theta to " 
			//	<< i << "/6 of full_theta_data" << std::endl;
			++ k;
		}
	}

	
	const int NHW_threads = N*output_H_ * output_W_;
	
	
	// compute out input_grid_data
	
	
	//////////////store data from U to input grid/////
	
		Compute_input_fromU<Dtype><<<CAFFE_GET_BLOCKS(NHW_threads), CAFFE_CUDA_NUM_THREADS>>>( 
			NHW_threads, U, input_grid_data, output_H_, output_W_, C);
	
		
	
		for (int i = 0; i < N; ++i) {

			Dtype* Output_address = output_grid_data + (output_H_ * output_W_ * 3) * i;
			Dtype* Input_address = input_grid_data + (output_H_ * output_W_ * 4) * i;

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, output_H_ * output_W_, 3, 4, (Dtype)1.,
			Input_address, full_theta_data + 12 * i, (Dtype)0., Output_address);
		}
				
	/////////////////////store data from output grid to V//////////////
	
		ComputeV_from_output<Dtype><<<CAFFE_GET_BLOCKS(NHW_threads), CAFFE_CUDA_NUM_THREADS>>>( 
			NHW_threads, V, output_grid_data, output_H_, output_W_, C);
								

}



template <typename Dtype>
__global__ void ColorTransformerBackwardGPU_dTheta(const int nthreads, Dtype* dTheta_t, const Dtype* dVV, const Dtype* UU, int output_H_, int output_W_, int C) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int i = index / (output_W_ * output_H_);

		int idx =  s * output_W_ + t;
		dTheta_t[(12 * i)*output_H_*output_W_ + idx] += dVV[i*output_H_*output_W_*C + idx] * UU[i*output_H_*output_W_*C + idx];
		dTheta_t[(12 * i + 1)*output_H_*output_W_ + idx] += dVV[i*output_H_*output_W_*C + idx] * UU[i*output_H_*output_W_*C + idx + output_H_*output_W_];
		dTheta_t[(12 * i + 2)*output_H_*output_W_ + idx] += dVV[i*output_H_*output_W_*C + idx] * UU[i*output_H_*output_W_*C + idx + 2 * output_H_*output_W_];
		dTheta_t[(12 * i + 3)*output_H_*output_W_ + idx] += dVV[i*output_H_*output_W_*C + idx];
					
					
		dTheta_t[(12 * i + 4)*output_H_*output_W_ + idx] += dVV[i*output_H_*output_W_*C + idx + output_H_*output_W_] * UU[i*output_H_*output_W_*C + idx];
		dTheta_t[(12 * i + 5)*output_H_*output_W_ + idx] += dVV[i*output_H_*output_W_*C + idx + output_H_*output_W_] * UU[i*output_H_*output_W_*C + idx + output_H_*output_W_];
		dTheta_t[(12 * i + 6)*output_H_*output_W_ + idx] += dVV[i*output_H_*output_W_*C + idx + output_H_*output_W_] * UU[i*output_H_*output_W_*C + idx + 2 * output_H_*output_W_];
		dTheta_t[(12 * i + 7)*output_H_*output_W_ + idx] += dVV[i*output_H_*output_W_*C + idx + output_H_*output_W_];
					
		dTheta_t[(12 * i + 8)*output_H_*output_W_ + idx] += dVV[i*output_H_*output_W_*C + idx + 2 * output_H_*output_W_] * UU[i*output_H_*output_W_*C + idx];
		dTheta_t[(12 * i + 9)*output_H_*output_W_ + idx] += dVV[i*output_H_*output_W_*C + idx + 2 * output_H_*output_W_] * UU[i*output_H_*output_W_*C + idx + output_H_*output_W_];
		dTheta_t[(12 * i + 10)*output_H_*output_W_ + idx] += dVV[i*output_H_*output_W_*C + idx + 2 * output_H_*output_W_] * UU[i*output_H_*output_W_*C + idx + 2 * output_H_*output_W_];
		dTheta_t[(12 * i + 11)*output_H_*output_W_ + idx] += dVV[i*output_H_*output_W_*C + idx + 2 * output_H_*output_W_];
		
	}
}


template <typename Dtype>
void TransformColorLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	string prefix = "TransformColorLayer::Backward_GPU::\t";

	const Dtype* dV = top[0]->gpu_diff();
	const Dtype* U = bottom[0]->gpu_data();
	Dtype* dTheta = bottom[1]->mutable_gpu_diff();

	Dtype* dTheta_tmp_diff = dTheta_tmp.mutable_gpu_diff();
	Dtype* dFull_theta = full_theta.mutable_gpu_diff();
	

	caffe_gpu_set(dTheta_tmp.count(), (Dtype)0., dTheta_tmp_diff);
	//caffe_gpu_set(bottom[1]->count(), (Dtype)0, dTheta);
	/////////tinh dtheta/////////////
	
	
	const int NHW_threads = N*output_H_ * output_W_;
	
	
		ColorTransformerBackwardGPU_dTheta<Dtype><<<CAFFE_GET_BLOCKS(NHW_threads),
			CAFFE_CUDA_NUM_THREADS>>>(NHW_threads, dTheta_tmp_diff, dV, U, output_H_, output_W_, C);
	


		Dtype* all_ones_2_data = all_ones_2.mutable_gpu_data();
		caffe_gpu_set(all_ones_2.count(), (Dtype)1., all_ones_2_data);

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, full_theta.count(), 1, output_H_ * output_W_,
			(Dtype)1., dTheta_tmp_diff, all_ones_2_data, (Dtype)0., dFull_theta);
	


		int k = 0;
		const int num_threads = N;
		for (int i = 0; i<12; ++i) {
			if (!is_pre_defined_theta[i]) {
				copy_values<Dtype> << <CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(num_threads,
					12, i, dFull_theta, 12 - pre_defined_count, k, dTheta);
				//std::cout << "Copying " << i << "/6 of dFull_theta to " << k << "/" << 
				//	6 - pre_defined_count << " of dTheta" << std::endl;
				++k;
			}
		}

	///////////////////END/////////////
	
	
	
	
}

INSTANTIATE_LAYER_GPU_FUNCS(TransformColorLayer);

}	// namespace caffe
