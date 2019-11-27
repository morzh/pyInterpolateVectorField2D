
#include "pybind11/pybind11.h"            // Pybind11 import to define Python bindings

#include "xtensor/xmath.hpp"              // xtensor import for the C++ universal functions
#include "xtensor/xtensor.hpp"              // xtensor import for the C++ universal functions
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor-blas/xlinalg.hpp"
#define FORCE_IMPORT_ARRAY                // numpy C api loading
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include <numeric>                        // Standard library import for std::accumulate
#include <pybind11/stl.h>

#include <array>
#include <iostream>


#define PI 3.14159265


namespace py = pybind11;

class InterpolateFlow{
public:


    void                    calc_kernelLandmarks    (const xt::pyarray<float> &lms_src, const xt::pyarray<float> &lms_dst, float l= 1e-4, float s= 3.5);
    xt::xarray<float>       calc_flowAtPoint        (const xt::xarray<float> &point, const xt::pyarray<float> &lms);
    xt::pyarray<float>      get_flow                (std::array<int, 2> min, std::array<int, 2> max, const xt::pyarray<float> &lms_src);
    void                    print();

private:
    void                    calc_vectorField        (const xt::pyarray<float> &lms_src, const xt::pyarray<float> &lms_dst);

    xt::xarray<float>       G1, beta, V1;
    float                   lambda=1e-4;
    float                   sigma=3.5;
};



/// Python interface
PYBIND11_MODULE(myFlow,m) {

    m.doc() = "moduile for building smooth image deformations from two sets of points";
    xt::import_numpy();

    py::class_<InterpolateFlow>(m, "InterpolateFlow")
            .def( py::init() )
            .def( "calc_kernelLandmarks", &InterpolateFlow::calc_kernelLandmarks )
            .def( "get_flow", &InterpolateFlow::get_flow )
            .def( "print", &InterpolateFlow::print);
}





/// class implementation

xt::pyarray<float> InterpolateFlow::get_flow(std::array<int, 2> min, std::array<int, 2> max, const xt::pyarray<float> &lms_src) {

    xt::pyarray<float> flow = xt::zeros<int>({max[0]-min[0], max[1]-min[1], 2});

    for (int i=min[0]; i<max[0]; ++i){
        for (int j=min[1]; j<max[1]; ++j){

            xt::xarray<float> pt = {static_cast<float>(i), static_cast<float>(j)};
            xt::xarray<float> v  = calc_flowAtPoint(pt, lms_src);

            flow(i,j,0) = v(0);
            flow(i,j,1) = v(1);
        }
    }

    return flow;
}

void InterpolateFlow::calc_kernelLandmarks(const xt::pyarray<float> &lms_src, const xt::pyarray<float> &lms_dst, float l, float s) {

    if ( lms_src.shape() != lms_dst.shape() || lms_src.size() ==0 || lms_src.dimension() != 2 || lms_src.size() < 6 || lms_src.shape()[1] != 2)
        std::cout << "input arrays should be of the same size" <<std::endl;

    calc_vectorField(lms_src, lms_dst);

    this->lambda    = l;
    this->sigma     = s;

    bool                debug           = false;
    unsigned long int   num_centroids   =  lms_src.shape()[0];
    xt::xarray<float>   G                = xt::xarray<float>::from_shape( {num_centroids, num_centroids});
    xt::xarray<float>   lambda_kronecker = xt::xarray<float>::from_shape( {num_centroids, num_centroids});

    G1.resize({num_centroids, num_centroids});
    G.fill(0.0);
    lambda_kronecker.fill(0.0);


    for( int i=0; i< num_centroids; ++i){
        for( int j=0; j<num_centroids; ++j){

            xt::xarray<float> c_i = xt::view(lms_src, i, xt::all());
            xt::xarray<float> c_j = xt::view(lms_src, j, xt::all());

            G(i,j) =  1.0 / (2*PI* std::pow(sigma, 2)) * std::exp(-xt::linalg::norm(c_i-c_j)  / (2*std::pow(sigma, 2) ));

            if (i == j)
                lambda_kronecker(i,j) = lambda;
        }
    }

    G1      = G + lambda_kronecker;
    beta    =  xt::linalg::dot(xt::linalg::inv(G1),  V1);
}

void InterpolateFlow::calc_vectorField(const xt::pyarray<float> &lms_src, const xt::pyarray<float> &lms_dst) {

    this->V1  = xt::xarray<float>::from_shape(lms_src.shape());

    for (int row=0; row<lms_src.shape()[0]; ++row){
        for (int col=0; col<2; ++col){

            V1(row, col) = lms_src(row, col) - lms_dst(row, col);
        }
    }
}

xt::xarray<float> InterpolateFlow::calc_flowAtPoint(const xt::xarray<float> &point, const xt::pyarray<float> &lms) {

//    Eigen::Vector2f v_r;
//    v_r.setZero();

    xt::xarray<float>   v_r = xt::xarray<float>::from_shape( {1,2});
    v_r.fill(0.0);

    for( int i=0; i<  beta.shape()[0]; ++i){

        xt::xarray<float> c_i     = xt::view(lms,  i, xt::all());;
        xt::xarray<float> beta_i  = xt::view(beta, i, xt::all());;

        v_r += beta_i / (2*PI* std::pow(sigma,2))   *  std::exp(-xt::linalg::norm(point-c_i)  / (2*std::pow(sigma, 2) ));
    }

    return  v_r;
}


void InterpolateFlow::print() {

    std::cout << "Module for interpolationg sparse vector field to a dense one. Variational problem used to  solved this task" << std::endl;
}
