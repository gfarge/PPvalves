! =============================================================================
! 
! Two fortran subroutines to do fast maths computations with purely tridiagonal
! matrices, and plug them into python using f2py. 
! 
! First subroutine solves the A.x = b matrix system, where A is tridiagonal, 
! using the Thomas algorithm.
!
! Second subroutine computes the matrix vector product A.x = p, 
! where A is tridiagonal.
!
! Author : Gaspard Farge (gfarge@gmail.com)
! Date written : 25 March 2021
! Last updated : 25 March 2021
!
! =============================================================================

subroutine solve(low, mid, up, d, x, N)

! -----------------------------------------------------------------------------
!
!  Solves Ax = b, where A is tridiagonal.
!
! -----------------------------------------------------------------------------
!
!  Parameters 
!  ==========
!  low - sub-diagonal (means it is the diagonal below the main diagonal)
!  mid - the main diagonal
!  up  - sup-diagonal (means it is the diagonal above the main diagonal)
!  d   - right part
!  N   - number of equations (length of x, d, or any other input vectors)
!
!  Returns
!  =======
!  x   - the solution (returns)
! 
! Algorithm adapted from : https://en.wikibooks.org/wiki/Algorithm_Implementation/Linear_Algebra/Tridiagonal_matrix_algorithm#Fortran_90  (only variable names changed)
!
! Note
! ====
! Padding should be applied: lower diagonal starts with a 0, upper diagonal
! finishes with a 0, so that they share their size with the main diagonal. This
! does not correspond to the convention for scipy.solve_banded().
!
! -----------------------------------------------------------------------------

  implicit none
  integer, parameter :: dp=kind(1.d0)

  ! Parameters
  integer, intent(in) :: N
  real(dp), dimension(N), intent(in) :: low,mid,up,d
  real(dp), dimension(N), intent(out) :: x
  
  ! Dummy variables
  real(dp), dimension(N) :: upc,dc
  real(dp) :: m
  integer i

! >> Instructions to wrap using numpy's f2py
!f2py intent(out) x

!f2py intent(in) up
!f2py intent(in) mid
!f2py intent(in) low
!f2py intent(in) d
!f2py intent(in) n

!f2py depend(n) x
!f2py depend(n) up
!f2py depend(n) mid
!f2py depend(n) low
!f2py depend(n) d

  ! >> Initialize up-copy and d-copy
  upc(1) = up(1) / mid(1)
  dc(1) = d(1) / mid(1)

  ! >> Solve for vectors up-copy and d-copy
  do i = 2, N
    m = mid(i)-upc(i-1)*low(i)
    upc(i) = up(i)/m
    dc(i) = (d(i)-dc(i-1)*low(i))/m
  end do

  ! >> Initialize x
  x(N) = dc(N)

  ! >> Solve for x from the vectors up-copy and d-copy
  do i = N-1, 1, -1
    x(i) = dc(i)-upc(i)*x(i+1)
  end do

end subroutine solve

! ============================================================================
! ============================================================================

subroutine prod(low, mid, up, b, p, N)
! -----------------------------------------------------------------------------
!
!  Computes A.b = p, where A is tridiagonal.
!
! -----------------------------------------------------------------------------
!
!  Parameters 
!  ==========
!  low - sub-diagonal (means it is the diagonal below the main diagonal)
!  mid - the main diagonal
!  up  - sup-diagonal (means it is the diagonal above the main diagonal)
!  b   - vector to multiply with
!  N   - number of equations (length of x, d, or any other input vectors)
!
!  Returns 
!  =======
!  p   - the product
! 
! Note
! ====
! Padding should be applied: lower diagonal starts with a 0, upper diagonal
! finishes with a 0, so that they share their size with the main diagonal. This
! does not correspond to the convention for scipy.solve_banded().
!
! -----------------------------------------------------------------------------
  implicit none
  integer, parameter :: dp=kind(1.d0)

  ! - Arguments
  integer, intent(in) :: N  ! size of vector
  real(dp), dimension(N), intent(in) :: up, mid, low, b
  real(dp), dimension(N), intent(out) :: p

  ! - Dummy variables
  integer :: ii   ! iteration

! >> Instruction to wrap with numpy's f2py
!f2py intent(out) p

!f2py intent(in) low
!f2py intent(in) mid
!f2py intent(in) up
!f2py intent(in) b
!f2py intent(in) n

!f2py depend(n) low
!f2py depend(n) mid
!f2py depend(n) up
!f2py depend(n) b
!f2py depend(n) p

  ! >> Top of vector
  p(1) = mid(1)*b(1) + up(1)*b(2)

  ! >> Middle of vector
  do ii=2,N-1
    p(ii) = low(ii)*b(ii-1) + mid(ii)*b(ii) + up(ii)*b(ii+1)
  end do

  ! >> Bottom of vector
  p(N) = low(N)*b(N-1) + mid(N)*b(N) 

end subroutine prod
