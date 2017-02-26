/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPSCORE_PIEACEWISE_POLYNOMIAL_HPP
#define ALPSCORE_PIEACEWISE_POLYNOMIAL_HPP

#include <complex>
#include <cmath>
#include <vector>

#include <boost/multi_array.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/typeof/typeof.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/multi_array.hpp>

#ifdef ALPS_HAVE_MPI
#include "mpi_bcast.hpp"
#endif

/**
 * @brief Class representing a pieacewise polynomial and utilities
 */
namespace alps {
    namespace gf {

        template<typename T, int k>
        class piecewise_polynomial;

        namespace detail {
            /**
             *
             * @tparam T double or std::complex<double>
             * @param a  scalar
             * @param b  scalar
             * @return   conj(a) * b
             */
            template<class T>
            typename boost::enable_if<boost::is_floating_point<T>, T>::type
            outer_product(T a, T b) {
                return a * b;
            }

            template<class T>
            std::complex<T>
            outer_product(const std::complex<T> &a, const std::complex<T> &b) {
                return std::conj(a) * b;
            }

            template<class T>
            typename boost::enable_if<boost::is_floating_point<T>, T>::type
            conjg(T a) {
                return a;
            }

            template<class T>
            std::complex<T>
            conjg(const std::complex<T> &a) {
                return std::conj(a);
            }
        }

/**
 * Class for representing a piecewise polynomial
 *   A function is represented by a polynomial in each section [x_n, x_{n+1}).
 */
        template<typename T, int k>
        class piecewise_polynomial {
        private:
            typedef boost::multi_array<T, 2> coefficient_type;

            template<typename TT, int kk>
            friend piecewise_polynomial<TT, kk>
            operator+(const piecewise_polynomial<TT, kk> &f1, const piecewise_polynomial<TT, kk> &f2);

            template<typename TT, int kk>
            friend piecewise_polynomial<TT, kk>
            operator-(const piecewise_polynomial<TT, kk> &f1, const piecewise_polynomial<TT, kk> &f2);

            template<typename TT, int kk>
            friend const piecewise_polynomial<TT, kk> operator*(TT scalar, const piecewise_polynomial<TT, kk> &pp);

            template<typename TT, int kk>
            friend
            class piecewise_polynomial;

            /// number of sections
            int n_sections_;

            /// edges of sections. The first and last elements should be -1 and 1, respectively.
            std::vector<double> section_edges_;

            /// expansion coefficients [s,l]
            /// The polynomial is represented as
            ///   \sum_{l=0}^k a_{s,l} (x - x_s)^l,
            /// where x_s is the left end point of the s-th section.
            coefficient_type coeff_;

            bool valid_;

            void check_range(double x) const {
                if (x < section_edges_[0] || x > section_edges_[section_edges_.size() - 1]) {
                    throw std::runtime_error("Give x is out of the range.");
                }
            }

            void check_validity() const {
                std::cout << "debug2 "<< valid_ << std::endl;
                if (!valid_) {
                    throw std::runtime_error("pieacewise_polynomial object is not properly constructed!");
                }
            }

            void set_validity() {
                valid_ = true;
                valid_ = valid_ && (n_sections_ >= 1);
                assert(valid_);
                valid_ = valid_ && (section_edges_.size() == n_sections_ + 1);
                assert(valid_);
                valid_ = valid_ && (coeff_.shape()[0] == n_sections_);
                assert(valid_);
                valid_ = valid_ && (coeff_.shape()[1] == k + 1);
                assert(valid_);
                for (int i = 0; i < n_sections_; ++i) {
                    valid_ = valid_ && (section_edges_[i] < section_edges_[i + 1]);
                }
                assert(valid_);
                std::cout << "debug "<< valid_ << std::endl;
            }

        public:
            piecewise_polynomial() : n_sections_(0), valid_(false) {};

            piecewise_polynomial(int n_section,
                                 const std::vector<double> &section_edges,
                                 const boost::multi_array<T, 2> &coeff) : n_sections_(section_edges.size() - 1),
                                                                          section_edges_(section_edges),
                                                                          coeff_(coeff), valid_(false) {
                set_validity();
            };

            /// Number of sections
            int num_sections() const {
                check_validity();
                return n_sections_;
            }

            inline double section_edge(int i) const {
                assert(i >= 0 && i < section_edges_.size());
                check_validity();
                return section_edges_[i];
            }

            const std::vector<double> &section_edges() const {
                check_validity();
                return section_edges_;
            }

            inline T coefficient(int i, int p) const {
                assert(i >= 0 && i < section_edges_.size());
                assert(p >= 0 && p <= k);
                check_validity();
                return coeff_[i][p];
            }

            /// Compute the value at x
            inline T compute_value(double x) const {
                check_validity();
                return compute_value(x, find_section(x));
            }

            /// Compute the value at x. x must be in the given section.
            inline T compute_value(double x, int section) const {
                check_validity();
                if (x < section_edges_[section] || (x != section_edges_.back() && x >= section_edges_[section + 1])) {
                    throw std::runtime_error("The given x is not in the given section.");
                }

                const double dx = x - section_edges_[section];
                T r = 0.0, x_pow = 1.0;
                for (int p = 0; p < k + 1; ++p) {
                    r += coeff_[section][p] * x_pow;
                    x_pow *= dx;
                }
                return r;
            }

            /// Find the section involving the given x
            int find_section(double x) const {
                check_validity();
                if (x == section_edges_[0]) {
                    return 0;
                } else if (x == section_edges_.back()) {
                    return coeff_.size() - 1;
                }

                std::vector<double>::const_iterator it =
                        std::upper_bound(section_edges_.begin(), section_edges_.end(), x);
                --it;
                return (&(*it) - &(section_edges_[0]));
            }

            /// Compute overlap <this | other> with complex conjugate
            template<class T2, int k2>
            T overlap(const piecewise_polynomial<T2, k2> &other) const {
                check_validity();
                if (section_edges_ != other.section_edges_) {
                    throw std::runtime_error("Not supported");
                }
                typedef BOOST_TYPEOF(static_cast<T>(1.0)*static_cast<T2>(1.0))  Tr;

                Tr r = 0.0;
                boost::array<double, k + k2 + 2> x_min_power, dx_power;

                for (int s = 0; s < n_sections_; ++s) {
                    dx_power[0] = 1.0;
                    const double dx = section_edges_[s + 1] - section_edges_[s];
                    for (int p = 1; p < dx_power.size(); ++p) {
                        dx_power[p] = dx * dx_power[p - 1];
                    }

                    for (int p = 0; p < k + 1; ++p) {
                        for (int p2 = 0; p2 < k2 + 1; ++p2) {
                            r += detail::outer_product((Tr) coeff_[s][p], (Tr) other.coeff_[s][p2])
                                 * dx_power[p + p2 + 1] / (p + p2 + 1.0);
                        }
                    }
                }
                return r;
            }

            bool operator==(const piecewise_polynomial<T, k> &other) const {
                return (n_sections_ == other.n_sections_) &&
                        (section_edges_ == other.section_edges_) &&
                                (coeff_ == other.coeff_);
            }

            void save(alps::hdf5::archive& ar, const std::string& path) const {
                check_validity();
                ar[path+"/k"] <<  k;
                ar[path+"/num_sections"] << num_sections();
                ar[path+"/section_edges"] << section_edges_;
                ar[path+"/coefficients"] << coeff_;
            }

            void load(alps::hdf5::archive& ar, const std::string& path) {
                int k_tmp;
                ar[path+"/k"] >>  k_tmp;
                if (k != k_tmp) {
                    throw std::runtime_error("Attempt to load data with different polynomial order k"+boost::lexical_cast<std::string>(k_tmp));
                }
                ar[path+"/num_sections"] >> n_sections_;
                ar[path+"/section_edges"] >> section_edges_;
                ar[path+"/coefficients"] >> coeff_;

                set_validity();
                check_validity();
            }

#ifdef ALPS_HAVE_MPI
            void broadcast(const alps::mpi::communicator& comm, int root)
            {
                using alps::mpi::broadcast;

                broadcast(comm, n_sections_, root);
                section_edges_.resize(n_sections_+1);
                broadcast(comm, &section_edges_[0], n_sections_+1, root);

                coeff_.resize(boost::extents[n_sections_][k+1]);
                broadcast(comm, coeff_.origin(), (k+1)*n_sections_, root);

                set_validity();
                check_validity();
            }
#endif

        };//class pieacewise_polynomial

/// Add piecewise_polynomial objects
        template<typename T, int k>
        piecewise_polynomial<T, k>
        operator+(const piecewise_polynomial<T, k> &f1, const piecewise_polynomial<T, k> &f2) {
            if (f1.section_edges_ != f2.section_edges_) {
                throw std::runtime_error("Cannot add two numerical functions with different sections!");
            }
            boost::multi_array<T, 2> coeff_sum(f1.coeff_);
            std::transform(
                    f1.coeff_.origin(), f1.coeff_.origin() + f1.coeff_.num_elements(),
                    f2.coeff_.origin(), coeff_sum.origin(),
                    std::plus<T>()

            );
            return piecewise_polynomial<T, k>(f1.num_sections(), f1.section_edges_, coeff_sum);
        }

/// Substract piecewise_polynomial objects
        template<typename T, int k>
        piecewise_polynomial<T, k>
        operator-(const piecewise_polynomial<T, k> &f1, const piecewise_polynomial<T, k> &f2) {
            if (f1.section_edges_ != f2.section_edges_) {
                throw std::runtime_error("Cannot add two numerical functions with different sections!");
            }
            boost::multi_array<T, 2> coeff_sum(f1.coeff_);
            std::transform(
                    f1.coeff_.origin(), f1.coeff_.origin() + f1.coeff_.num_elements(),
                    f2.coeff_.origin(), coeff_sum.origin(),
                    std::minus<T>()

            );
            return piecewise_polynomial<T, k>(f1.num_sections(), f1.section_edges_, coeff_sum);
        }

/// Multiply piecewise_polynomial by a scalar
        template<typename T, int k>
        const piecewise_polynomial<T, k> operator*(T scalar, const piecewise_polynomial<T, k> &pp) {
            piecewise_polynomial<T, k> pp_copy(pp);
            std::transform(
                    pp_copy.coeff_.origin(), pp_copy.coeff_.origin() + pp_copy.coeff_.num_elements(),
                    pp_copy.coeff_.origin(), std::bind1st(std::multiplies<T>(), scalar)

            );
            return pp_copy;
        }

/// Gram-Schmidt orthonormalization
        template<typename T, int k>
        void orthonormalize(std::vector<piecewise_polynomial<T, k> > &pps) {
            typedef piecewise_polynomial<T, k> pp_type;

            for (int l = 0; l < pps.size(); ++l) {
                pp_type pp_new(pps[l]);
                for (int l2 = 0; l2 < l; ++l2) {
                    const T overlap = pps[l2].overlap(pps[l]);
                    pp_new = pp_new - overlap * pps[l2];
                }
                double norm = pp_new.overlap(pp_new);
                pps[l] = (1.0 / std::sqrt(norm)) * pp_new;
            }
        }



    }
}

#endif //ALPSCORE_PIEACEWISE_POLYNOMIAL_HPP
