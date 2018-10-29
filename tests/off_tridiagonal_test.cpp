#include <catch2/catch.hpp>

#include <vector>
#include <eigen3/Eigen/Dense>

#include <tridiagonal/off_tridiagonal.hpp>

using std::vector;

using Eigen::Matrix;
using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector3d;
using Eigen::VectorXd;

using tridiagonal::solve_off_tridiagonal;

TEST_CASE("Off tridiagonal solver tests", "[offtridiagonal]") {

    SECTION("1x1 matrix system"){
        Matrix2d M; M << 1, 2, 3, 4;
        Vector2d D; D << 5, 6;

        vector<Matrix2d> lower_diagonal = {};
        vector<Matrix2d> diagonal = {M};
        vector<Matrix2d> upper_diagonal = {};
        vector<Vector2d> rhs = {D};

        vector<Vector2d> solution =
                solve_off_tridiagonal(lower_diagonal, diagonal, upper_diagonal, rhs);
        REQUIRE(solution.size() == 1);

        Matrix<double, 2, 1> eigen_solution = M.fullPivHouseholderQr().solve(D);
        REQUIRE(solution[0].isApprox(eigen_solution));
    }

    SECTION("2x2 matrix system"){
        Matrix2d M; M << 1, 2, 3, 4;
        Matrix<double, 2, 1> D; D << 5, 6;

        Matrix<double, 1, 1> a1, b0, b1, c0, d0, d1;
        a1 = M.block(1, 0, 1, 1);
        b0 = M.block(0, 0, 1, 1);
        b1 = M.block(1, 1, 1, 1);
        c0 = M.block(0, 1, 1, 1);

        d0 = D.block(0, 0, 1, 1);
        d1 = D.block(1, 0, 1, 1);

        vector<Matrix<double, 1, 1>> lower_diagonal = {a1};
        vector<Matrix<double, 1, 1>> diagonal = {b0, b1};
        vector<Matrix<double, 1, 1>> upper_diagonal = {c0};
        vector<Matrix<double, 1, 1>> rhs = {d0, d1};

        vector<Matrix<double, 1, 1>> solution =
                solve_off_tridiagonal(lower_diagonal, diagonal, upper_diagonal, rhs);
        REQUIRE(solution.size() == 2);

        Vector2d bezier_solution;
        bezier_solution << solution[0](0,0), solution[1](0,0);

        Vector2d eigen_solution = M.fullPivHouseholderQr().solve(D);

        REQUIRE(bezier_solution.isApprox(eigen_solution));
    }

    SECTION("2x2 block matrix (different sizes) system"){
        Matrix<double, 5, 5> M;
        M <<    1, 2,       3, -4, 2,
                -1, 6,      7, 8, 5,

                9, 2,       11, -12, 9,
                13, 14,     5, 16, 1,
                3, 2,       0, -1, 3;

        Matrix<double, 5, 2> D;
        D << 17, 18, 19, 20, 2,
             23, 43, -2, 5, 7;

        Matrix<double, 3, 2> a1 = M.block(2, 0, 3, 2);

        Matrix2d b0 = M.block(0, 0, 2, 2);
        Matrix3d b1 = M.block(2, 2, 3, 3);

        Matrix<double, 2, 3> c0 = M.block(0, 2, 2, 3);

        Matrix<double, 2, 2> d0 = D.block(0, 0, 2, 2);
        Matrix<double, 3, 2> d1 = D.block(2, 0, 3, 2);

        vector<MatrixXd> lower_diagonal = {a1};
        vector<MatrixXd> diagonal = {b0, b1};
        vector<MatrixXd> upper_diagonal = {c0};
        vector<MatrixXd> rhs = {d0, d1};

        vector<MatrixXd> solution =
                solve_off_tridiagonal(lower_diagonal, diagonal, upper_diagonal, rhs);
        REQUIRE(solution.size() == 2);

        Matrix<double, 5, 2> bezier_solution;
        bezier_solution.block(0, 0, 2, 2) = solution[0];
        bezier_solution.block(2, 0, 3, 2) = solution[1];

        Matrix<double, 5, 2> eigen_solution = M.fullPivHouseholderQr().solve(D);

        REQUIRE(bezier_solution.isApprox(eigen_solution));

    }

    SECTION("3x3 block matrix (2x2) system"){

        Matrix2d a0, a1, a2, b0, b1, b2, c0, c1, c2;
        Vector2d d0, d1, d2;

        Matrix<double, 6, 6> M;

        M <<    0, 3,  1, 2,  2, -1,
                2, 1,  4, 5,  5, 9,

                6, 7,  1, 2,  4, 5,
                1, 3,  9, 0,  2, 3,

                2, -1,  2, 1,  5, 2,
                4, 3,  3, 7,  9, 3;


        Matrix<double, 6, 1> D;
        D << 0, 2, 3, 1, -9, 2;

        b0 = M.block(0, 0, 2, 2);
        b1 = M.block(2, 2, 2, 2);
        b2 = M.block(4, 4, 2, 2);

        c0 = M.block(0, 2, 2, 2);
        c1 = M.block(2, 4, 2, 2);
        c2 = M.block(4, 0, 2, 2);

        a0 = M.block(0, 4, 2, 2);
        a1 = M.block(2, 0, 2, 2);
        a2 = M.block(4, 2, 2, 2);

        d0 = D.block(0, 0, 2, 1);
        d1 = D.block(2, 0, 2, 1);
        d2 = D.block(4, 0, 2, 1);

        vector<Matrix2d> diagonal_elements = {b0, b1, b2};
        vector<Matrix2d> lower_diagonal_elements = {a0, a1, a2};
        vector<Matrix2d> upper_diagonal_elements = {c0, c1, c2};
        vector<Vector2d> right_hand_side = {d0, d1, d2};

        vector<Vector2d> solution =
                solve_off_tridiagonal(lower_diagonal_elements,
                                                   diagonal_elements,
                                                   upper_diagonal_elements,
                                                   right_hand_side);
        REQUIRE(solution.size() == 3);

        Matrix<double, 6, 1> bezier_solution;
        bezier_solution.block(0, 0, 2, 1) = solution[0];
        bezier_solution.block(2, 0, 2, 1) = solution[1];
        bezier_solution.block(4, 0, 2, 1) = solution[2];

        Matrix<double, 6, 1> eigen_solution = M.fullPivHouseholderQr().solve(D);

        REQUIRE(bezier_solution.isApprox(eigen_solution));
    }

    SECTION("4x4 matrix system") {
        Matrix<double, 4, 4> M;
        M << 1.1,  2.2,  0,   -2.3,
             3.5,  4.1,  5.9,  0,
             0,    2.0, -5.3,  3.6,
            -1.5,  0,    6.2,  7.1;

        Matrix<double, 4, 1> D;
        D << 8.4, 9.3, 10.1, 11.8;

        Matrix<double, 1, 1> a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3;
        a0 = M.block(0, 3, 1, 1);
        a1 = M.block(1, 0, 1, 1);
        a2 = M.block(2, 1, 1, 1);
        a3 = M.block(3, 2, 1, 1);

        b0 = M.block(0, 0, 1, 1);
        b1 = M.block(1, 1, 1, 1);
        b2 = M.block(2, 2, 1, 1);
        b3 = M.block(3, 3, 1, 1);

        c0 = M.block(0, 1, 1, 1);
        c1 = M.block(1, 2, 1, 1);
        c2 = M.block(2, 3, 1, 1);
        c3 = M.block(3, 0, 1, 1);

        d0 = D.block(0, 0, 1, 1);
        d1 = D.block(1, 0, 1, 1);
        d2 = D.block(2, 0, 1, 1);
        d3 = D.block(3, 0, 1, 1);

        vector<Matrix<double, 1, 1>> lower_diagonal = {a0, a1, a2, a3};
        vector<Matrix<double, 1, 1>> diagonal = {b0, b1, b2, b3};
        vector<Matrix<double, 1, 1>> upper_diagonal = {c0, c1, c2, c3};
        vector<Matrix<double, 1, 1>> rhs = {d0, d1, d2, d3};

        vector<Matrix<double, 1, 1>> solution =
                solve_off_tridiagonal(lower_diagonal, diagonal, upper_diagonal, rhs);
        REQUIRE(solution.size() == 4);

        Matrix<double, 4, 1> bezier_solution;
        bezier_solution << solution[0](0, 0), solution[1](0, 0), solution[2](0, 0), solution[3](0, 0);

        Matrix<double, 4, 1> eigen_solution = M.fullPivHouseholderQr().solve(D);

        REQUIRE(bezier_solution.isApprox(eigen_solution));
    }


    SECTION("tests invalid arguments"){
        MatrixXd a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1, d2;

        Eigen::Matrix<double, 6, 6> M;

        M <<    0, 3,  1, 2,  2, -1,
                2, 1,  4, 5,  5, 9,

                6, 7,  1, 2,  4, 5,
                1, 3,  9, 0,  2, 3,

                2, -1,  2, 1,  5, 2,
                4, 3,  3, 7,  9, 3;


        Eigen::Matrix<double, 6, 1> D;
        D << 0, 2, 3, 1, -9, 2;

        b0 = M.block(0, 0, 2, 2);
        b1 = M.block(2, 2, 2, 2);
        b2 = M.block(4, 4, 2, 2);

        c0 = M.block(0, 2, 2, 2);
        c1 = M.block(2, 4, 2, 2);
        c2 = M.block(4, 0, 2, 2);

        a0 = M.block(0, 4, 2, 2);
        a1 = M.block(2, 0, 2, 2);
        a2 = M.block(4, 2, 2, 2);

        d0 = D.block(0, 0, 2, 1);
        d1 = D.block(2, 0, 2, 1);
        d2 = D.block(4, 0, 2, 1);

        vector<MatrixXd> lower_diagonal, diagonal, upper_diagonal, rhs;

        // Check invalid number of matrices
        lower_diagonal = {a0, a1}; diagonal = {b0, b1, b2}; upper_diagonal = {c0, c1, c2}; rhs = {d0, d1, d2};
        REQUIRE_THROWS_AS(solve_off_tridiagonal(lower_diagonal, diagonal, upper_diagonal, rhs), std::invalid_argument);

        lower_diagonal = {a0, a1, a2}; diagonal = {b0, b1}; upper_diagonal = {c0, c1, c2}; rhs = {d0, d1, d2};
        REQUIRE_THROWS_AS(solve_off_tridiagonal(lower_diagonal, diagonal, upper_diagonal, rhs), std::invalid_argument);

        lower_diagonal = {a0, a1, a2}; diagonal = {b0, b1, b2}; upper_diagonal = {c0, c1}; rhs = {d0, d1, d2};
        REQUIRE_THROWS_AS(solve_off_tridiagonal(lower_diagonal, diagonal, upper_diagonal, rhs), std::invalid_argument);

        lower_diagonal = {a0, a1, a2}; diagonal = {b0, b1, b2}; upper_diagonal = {c0, c1, c2}; rhs = {d0, d1};
        REQUIRE_THROWS_AS(solve_off_tridiagonal(lower_diagonal, diagonal, upper_diagonal, rhs), std::invalid_argument);

        // Check invalid dimension of matrices
        lower_diagonal = {a0, a1.block(0, 0, 1, 1), a2}; diagonal = {b0, b1, b2}; upper_diagonal = {c0, c1, c2}; rhs = {d0, d1, d2};
        REQUIRE_THROWS_AS(solve_off_tridiagonal(lower_diagonal, diagonal, upper_diagonal, rhs), std::invalid_argument);

        lower_diagonal = {a0, a1, a2}; diagonal = {b0, b1, b2.block(0, 0, 1, 1)}; upper_diagonal = {c0, c1, c2}; rhs = {d0, d1, d2};
        REQUIRE_THROWS_AS(solve_off_tridiagonal(lower_diagonal, diagonal, upper_diagonal, rhs), std::invalid_argument);

        lower_diagonal = {a0, a1, a2}; diagonal = {b0, b1, b2}; upper_diagonal = {c0, c1, c2.block(0, 0, 1, 1)}; rhs = {d0, d1, d2};
        REQUIRE_THROWS_AS(solve_off_tridiagonal(lower_diagonal, diagonal, upper_diagonal, rhs), std::invalid_argument);

        lower_diagonal = {a0, a1, a2}; diagonal = {b0, b1, b2}; upper_diagonal = {c0, c1, c2}; rhs = {d0.block(0, 0, 1, 1), d1, d2};
        REQUIRE_THROWS_AS(solve_off_tridiagonal(lower_diagonal, diagonal, upper_diagonal, rhs), std::invalid_argument);

        lower_diagonal = {a0, a1, a2}; diagonal = {b0, b1, b2}; upper_diagonal = {c0, c1, c2}; rhs = {d0, d1, d2};
        REQUIRE_NOTHROW(solve_off_tridiagonal(lower_diagonal, diagonal, upper_diagonal, rhs));
    }
}
