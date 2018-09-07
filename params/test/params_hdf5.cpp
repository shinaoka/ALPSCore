/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params_hdf5.cpp

    @brief Tests saving/loading of parameters
*/

#include "./params_test_support.hpp"

#include <iostream>

using alps::params;
namespace ah5=alps::hdf5;

namespace test_data {
    static const char inifile_content[]=
        "my_bool=true\n"
        "my_int=1234\n"
        "my_string=ABC\n"
        "my_double=12.75\n"
        ;

}

class ParamsTest : public ::testing::Test {
  protected:
    ParamsAndFile params_and_file_;
    params& par_;
    alps::testing::unique_file file_;
  public:
    ParamsTest() : params_and_file_(::test_data::inifile_content),
                   par_(*params_and_file_.get_params_ptr()),
                   file_("params_hdf5_test.h5.", alps::testing::unique_file::REMOVE_AFTER/*KEEP_AFTER*/)

    {   }
};

TEST_F(ParamsTest, saveLoad) {
    arg_holder args;
    args.add("some=something");
    params p_other(args.argc(), args.argv());
    p_other["another_int"]=9999;

    par_.define<int>("my_int", "Integer param");
    par_.define<double>("my_double", 0.00, "Double param");

    {
        ah5::archive ar(file_.name(), "w");
        ar["params"] << par_;
    }

    {
        ah5::archive ar(file_.name(), "r");
        ar["params"] >> p_other;
    }

    EXPECT_FALSE(p_other.exists("another_int"));
    EXPECT_EQ(par_, p_other);

    EXPECT_TRUE(p_other.define<std::string>("my_string", "", "String param").ok());
    EXPECT_EQ("ABC", p_other["my_string"].as<std::string>());

    EXPECT_FALSE(p_other.is_restored());
    EXPECT_ANY_THROW(p_other.get_archive_name());

    EXPECT_EQ(par_.get_argv0(), p_other.get_argv0());
    EXPECT_EQ(par_.get_ini_name_count(), p_other.get_ini_name_count());
    for (int i=0; i<par_.get_ini_name_count(); ++i) {
        EXPECT_EQ(par_.get_ini_name(i), p_other.get_ini_name(i));
    }
}

TEST_F(ParamsTest, h5Ctor) {
    {
        ah5::archive ar(file_.name(), "w");
        ar["/parameters"] << par_;
    }
    arg_holder args;
    args.add(file_.name());

    params p_new(args.argc(), args.argv());

    EXPECT_EQ(par_, p_new);

    EXPECT_FALSE(par_.is_restored());
    EXPECT_ANY_THROW(par_.get_archive_name());

    EXPECT_TRUE(p_new.is_restored());
    EXPECT_EQ(file_.name(), p_new.get_archive_name());
}

// struct define_visitor {
//     params& p_;
//     const std::string& name_;
//     const std::string& descr_;

//     define_visitor(params& p, const std::string& name, const std::string& descr): p_(p), name_(name), descr_(descr) {}
//     typedef void result_type;
//     template <typename T>
//     void operator()(const T& val) {
//         p_.define<T>(name_, descr_, val);
//     }
// };

TEST_F(ParamsTest, h5CtorOverride) {
    {
        ah5::archive ar(file_.name(), "w");
        par_
            .define<bool>("my_bool","Boolean")
            .define<int>("my_int", "Integer")
            .define<std::string>("my_string", "String")
            .define<double>("my_double", "Double");
        ar["/parameters"] << par_;
    }
    arg_holder args;
    args.add(file_.name()).add("my_int=7777");

    params p_new(args.argc(), args.argv());

    // params p_loaded(1, args.argv());
    // params p_new(args.argc()-1, args.argv()+1);
    // for (auto it=p_loaded.begin(), fin=p_loaded.end(); it!=fin; ++it) {
    //     apply_visitor(define_visitor(p_new, fin->name(), const std::string &descr)
    // }


    p_new.define<bool>("my_bool","Boolean")
        .define<int>("my_int", "Integer")
        .define<std::string>("my_string", "String")
        .define<double>("my_double", "Double");
    EXPECT_EQ(true, p_new["my_bool"]);
    EXPECT_EQ("ABC", p_new["my_string"]);
    EXPECT_EQ(12.75, p_new["my_double"]);
    EXPECT_EQ(7777, p_new["my_int"]);
}

TEST_F(ParamsTest, h5CtorNotFirstArgument) {
    {
        ah5::archive ar(file_.name(), "w");
        ar["/parameters"] << par_;
    }
    arg_holder args;
    args.add("some=something").add(file_.name());

    EXPECT_ANY_THROW(params p_new(args.argc(), args.argv()));
}
