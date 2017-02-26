/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "gtest/gtest.h"
#include "alps/gf/piecewise_polynomial.hpp"
#include <boost/filesystem/operations.hpp>

TEST(PiecewisePolynomial, Orthogonalization) {
    typedef double Scalar;
    const int n_section = 10, k = 8, n_basis = 3;
    typedef alps::gf::piecewise_polynomial<Scalar,k> pp_type;

    std::vector<double> section_edges(n_section+1);
    boost::multi_array<Scalar,3> coeff(boost::extents[n_basis][n_section][k+1]);

    for (int s = 0; s < n_section + 1; ++s) {
        section_edges[s] = s*2.0/n_section - 1.0;
    }
    section_edges[0] = -1.0;
    section_edges[n_section] = 1.0;

    std::vector<pp_type> nfunctions;

    // x^0, x^1, x^2, ...
    for (int n = 0; n < n_basis; ++ n) {
        boost::multi_array<Scalar,2> coeff(boost::extents[n_section][k+1]);
        std::fill(coeff.origin(), coeff.origin()+coeff.num_elements(), 0.0);

        for (int s = 0; s < n_section; ++s) {
            double rtmp = 1.0;
            for (int l = 0; l < k + 1; ++l) {
                if (n - l < 0) {
                    break;
                }
                if (l > 0) {
                    rtmp /= l;
                    rtmp *= n + 1 - l;
                }
                coeff[s][l] = rtmp * std::pow(section_edges[s], n-l);
            }
        }

        nfunctions.push_back(pp_type(n_section, section_edges, coeff));
    }

    // Check if correctly constructed
    double x = 0.9;
    for (int n = 0; n < n_basis; ++ n) {
        EXPECT_NEAR(nfunctions[n].compute_value(x), std::pow(x, n), 1e-8);
    }

    // Check overlap
    for (int n = 0; n < n_basis; ++ n) {
        for (int m = 0; m < n_basis; ++ m) {
            EXPECT_NEAR(nfunctions[n].overlap(nfunctions[m]), (std::pow(1.0,n+m+1)-std::pow(-1.0,n+m+1))/(n+m+1), 1e-8);
        }
    }


    // Check plus and minus
    for (int n = 0; n < n_basis; ++ n) {
        EXPECT_NEAR(4 * nfunctions[n].compute_value(x), (4.0*nfunctions[n]).compute_value(x), 1e-8);
        for (int m = 0; m < n_basis; ++m) {
            EXPECT_NEAR(nfunctions[n].compute_value(x) + nfunctions[m].compute_value(x),
                        (nfunctions[n] + nfunctions[m]).compute_value(x), 1e-8);
            EXPECT_NEAR(nfunctions[n].compute_value(x) - nfunctions[m].compute_value(x),
                        (nfunctions[n] - nfunctions[m]).compute_value(x), 1e-8);
        }
    }

    alps::gf::orthonormalize(nfunctions);
    for (int n = 0; n < n_basis; ++ n) {
        for (int m = 0; m < n_basis; ++m) {
            EXPECT_NEAR(nfunctions[n].overlap(nfunctions[m]),
                        n == m ? 1.0 : 0.0,
                        1e-8
            );
        }
    }

    //l = 0 should be x
    EXPECT_NEAR(nfunctions[1].compute_value(x) * std::sqrt(2.0/3.0), x, 1E-8);
}

TEST(PiecewisePolynomial, SaveLoad) {
    const int n_section = 2, k = 3;
    typedef double Scalar;
    typedef alps::gf::piecewise_polynomial<Scalar,k> pp_type;

    std::vector<double> section_edges(n_section+1);
    section_edges[0] = -1.0;
    section_edges[1] =  0.0;
    section_edges[2] =  1.0;
    boost::multi_array<Scalar,2> coeff(boost::extents[n_section][k+1]);
    std::fill(coeff.origin(), coeff.origin()+coeff.num_elements(), 0.0);

    pp_type p(n_section, section_edges, coeff), p2;
    {
        alps::hdf5::archive oar("pp.h5","w");
        p.save(oar,"/pp");

    }

    {
        alps::hdf5::archive iar("pp.h5");
        p2.load(iar,"/pp");
    }
    boost::filesystem::remove("pp.h5");

    ASSERT_TRUE(p == p2);
}


/*
TEST(Mesh,SwapLegendre) {
    alps::gf::legendre_mesh mesh_1(5.0, 20, alps::gf::statistics::FERMIONIC);
    alps::gf::legendre_mesh mesh_1r(mesh_1);
    alps::gf::legendre_mesh mesh_2(10.0, 40, alps::gf::statistics::BOSONIC);
    alps::gf::legendre_mesh mesh_2r(mesh_2);

    mesh_1.swap(mesh_2);
    EXPECT_EQ(mesh_1, mesh_2r);
    EXPECT_EQ(mesh_2, mesh_1r);
}

TEST(Mesh,PrintMatsubaraMeshHeader) {
    double beta=5.;
    int n=20;
    {
        std::stringstream header_line;
        header_line << "# MATSUBARA mesh: N: "<<n<<" beta: "<<beta<<" statistics: "<<"FERMIONIC"<<" POSITIVE_ONLY"<<std::endl;
        alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_ONLY> mesh1(beta, n);
        std::stringstream header_line_from_mesh;
        header_line_from_mesh << mesh1;
        EXPECT_EQ(header_line.str(), header_line_from_mesh.str());
    }
    {
        std::stringstream header_line;
        header_line << "# MATSUBARA mesh: N: "<<n<<" beta: "<<beta<<" statistics: "<<"FERMIONIC"<<" POSITIVE_NEGATIVE"<<std::endl;

}
 */
