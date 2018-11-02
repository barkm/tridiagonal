#ifndef TRIDIAGONAL_TRIDIAGONAL_HPP
#define TRIDIAGONAL_TRIDIAGONAL_HPP

#include <vector>
#include <type_traits>

#include "utils.hpp"

namespace tridiagonal {

    /**
     * Solve tri-diagonal block matrix system on the form
     *
     * |  B_0  C_0  0    0    0     ...     0     | | X_0   |     | D_0   |
     * |  A_0  B_1  C_1  0    0     ...     0     | | X_1   |     | D_1   |
     * |  0    A_1  B_2  C_2  0     ...     0     | | X_2   |     | D_2   |
     * |                                          | |       |     |       |
     * |  .           .    .    .           .     | |  .    |  =  |  .    |
     * |  .             .    .    .         .     | |  .    |     |  .    |
     * |  .               .    .    .       .     | |  .    |     |  .    |
     * |                                          | |       |     |       |
     * |                      A_n-3  B_n-2  C_n-2 | | X_n-2 |     | D_n-2 |
     * |  0                   0       A_n-2 B_n-1 | | X_n-1 |     | D_n-1 |
     *
     * using Thomas algorithm (https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm).
     *
     * @param lower_diagonal : list of matrices on the lower diagonal {A_1, ..., A_n-1}
     * @param diagonal : list of matrices on the diagonal {B_0, ..., B_n-1}
     * @param upper_diagonal : list of matrices on the upper diagonal {C_0, ..., C_n-2}
     * @param rhs : list of matrices on the right and side {D_0, ..., D_n-1}
     * @return list of solutions {X_0, ..., X_n-1}
     */
    template <typename A, typename B, typename C, typename D,
            typename std::enable_if<!is_dynamic<A>::value, int>::type = 0,
            typename std::enable_if<!is_dynamic<B>::value, int>::type = 0,
            typename std::enable_if<!is_dynamic<C>::value, int>::type = 0,
            typename std::enable_if<!is_dynamic<D>::value, int>::type = 0,
            typename std::enable_if<A::RowsAtCompileTime == A::ColsAtCompileTime, int>::type = 0,
            typename std::enable_if<A::ColsAtCompileTime == B::ColsAtCompileTime, int>::type = 0,
            typename std::enable_if<A::RowsAtCompileTime == C::RowsAtCompileTime, int>::type = 0,
            typename std::enable_if<A::RowsAtCompileTime == D::RowsAtCompileTime, int>::type = 0
    >
    vector<D> solve_tridiagonal(std::vector<A> const & lower_diagonal,
                                std::vector<B> const & diagonal,
                                std::vector<C> const & upper_diagonal,
                                std::vector<D> const & rhs);


    template <typename A, typename B, typename C, typename D,
            typename std::enable_if<is_dynamic<A>::value or
                                    is_dynamic<B>::value or
                                    is_dynamic<C>::value or
                                    is_dynamic<D>::value, int>::type = 0
    >
    vector<D> solve_tridiagonal(vector<A> const & lower_diagonal,
                                vector<B> const & diagonal,
                                vector<C> const & upper_diagonal,
                                vector<D> const & rhs);

}

#include "tridiagonal_solver.cpp"


#endif // TRIDIAGONAL_TRIDIAGONAL_H