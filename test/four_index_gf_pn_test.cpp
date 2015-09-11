#include "gtest/gtest.h"
#include "alps/gf/gf.hpp"

/// This generates some "outside" data to fill the mesh: 4 2-d points
inline alps::gf::momentum_index_mesh::container_type get_data_for_momentum_mesh()
{
    alps::gf::momentum_index_mesh::container_type points(boost::extents[4][2]);
    points[0][0]=0; points[0][1]=0; 
    points[1][0]=M_PI; points[1][1]=M_PI;
    points[2][0]=M_PI; points[2][1]=0; 
    points[3][0]=0; points[3][1]=M_PI;

    return points;
}

class TestGFM : public ::testing::Test
{
  public:
    const double beta;
    const int nsites;
    const int nfreq ;
    const int nspins;

    typedef alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_ONLY> matsubara_meshp_type;
    typedef alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_NEGATIVE> matsubara_meshn_type;

    typedef matsubara_meshp_type::index_type matsubara_indexp_type;
    typedef matsubara_meshn_type::index_type matsubara_indexn_type;

    typedef alps::gf::four_index_gf<std::complex<double>,
                          matsubara_meshp_type,
                          alps::gf::momentum_index_mesh,
                          alps::gf::momentum_index_mesh,
                          alps::gf::index_mesh> gfp_type;

    typedef alps::gf::four_index_gf<std::complex<double>,
                          matsubara_meshn_type,
                          alps::gf::momentum_index_mesh,
                          alps::gf::momentum_index_mesh,
                          alps::gf::index_mesh> gfn_type;

    gfp_type gf1;
    gfn_type gf2;

    TestGFM():beta(10), nsites(4), nfreq(10), nspins(2),
             gf1(matsubara_meshp_type(beta,nfreq),
                alps::gf::momentum_index_mesh(get_data_for_momentum_mesh()),
                alps::gf::momentum_index_mesh(get_data_for_momentum_mesh()),
                 alps::gf::index_mesh(nspins)),
             gf2(matsubara_meshn_type(beta,nfreq),
                alps::gf::momentum_index_mesh(get_data_for_momentum_mesh()),
                alps::gf::momentum_index_mesh(get_data_for_momentum_mesh()),
                 alps::gf::index_mesh(nspins))
              {}
        
};

TEST_F(TestGFM,PositiveNegative)
{
    alps::gf::momentum_index i; i=2;
    alps::gf::momentum_index j=alps::gf::momentum_index(3);
    alps::gf::index sigma(1);

    matsubara_indexp_type omega; omega=4;
    
    gf1(omega, i,j,sigma)=std::complex<double>(3,4);
    std::complex<double> x=gf1(omega,i,j,sigma);

    EXPECT_EQ(3,x.real());
    EXPECT_EQ(4,x.imag());

    matsubara_indexn_type omega2; omega2=-4;
    gf2(omega2, i,j,sigma)=std::complex<double>(3,4);
    x=gf2(omega2,i,j,sigma);

    EXPECT_EQ(3,x.real());
    EXPECT_EQ(4,x.imag());
}
