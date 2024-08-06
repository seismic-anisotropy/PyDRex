! Fortran 2003 miscellaneous utilities.
module misc
    use iso_fortran_env, only: real64
    use ieee_arithmetic, only: ieee_value, ieee_positive_inf
    implicit none
    contains

function argsort(array)
    ! Return indices that sort an array (increasing).
    real(kind=real64), intent(in), dimension(:) :: array

    real(kind=real64), dimension(size(array)) :: array_copy, array_sorted, argsort
    real(kind=real64) :: inf
    integer :: i, j, len
    inf = ieee_value(inf, ieee_positive_inf)

    array_copy = array
    array_sorted = array
    call quicksort_nr(array_sorted)

    len = size(array)
    do i = 1, len
        do j = 1, len
            if (array_sorted(i) == array_copy(j)) then
                argsort(j) = i
                array_copy(j) = inf
            end if
        end do
    end do
end function argsort

subroutine quicksort_nr(array)
    ! In-place quicksort implementation.
    ! Adapted from: https://www.mjr19.org.uk/IT/sorts/
    ! This version maintains its own stack, to avoid needing to call itself recursively.
    ! By always pushing the larger "half" to the stack,
    ! and moving directly to calculate the smaller "half",
    ! it can guarantee that the stack needs no more than log_2(N) entries.
    real(kind=real64), intent(inout), dimension(:) :: array
    real(kind=real64) :: temp, pivot
    integer :: i, j, left, right, low, high
    ! If your compiler lacks storage_size(), replace storage_size(i) with 64.
    integer :: stack(2, storage_size(i)), stack_ptr

    low = 1
    high = size(array)
    stack_ptr = 1

    do
        if (high - low < 50) then  ! Use insertion sort on small arrays.
            do i = low + 1, high
                temp = array(i)
                do j = i - 1, low, -1
                    if (array(j) < temp) then
                        exit
                    end if
                    array(j+1) = array(j)
                end do
                array(j+1) = temp
            end do
            ! Now pop from stack.
            if (stack_ptr == 1) then
                return
            end if
            stack_ptr = stack_ptr - 1
            low = stack(1, stack_ptr)
            high = stack(2, stack_ptr)
            cycle
        end if

        ! Find median of three pivot and place sentinels at first and last elements.
        temp = array((low+high)/2)
        array((low+high)/2) = array(low+1)
        if (temp > array(high)) then
            array(low+1) = array(high)
            array(high) = temp
        else
            array(low+1) = temp
        end if
        if (array(low) > array(high)) then
            temp = array(low)
            array(low) = array(high)
            array(high) = temp
        end if
        if (array(low) > array(low+1)) then
            temp = array(low)
            array(low) = array(low+1)
            array(low+1) = temp
        end if
        pivot = array(low+1)

        left = low+2
        right = high-1
        do
            do while(array(left) < pivot)
                left = left+1
            end do
            do while(array(right) > pivot)
                right = right-1
            end do
            if (left >= right) then
                exit
            end if
            temp = array(left)
            array(left) = array(right)
            array(right) = temp
            left = left+1
            right = right-1
        end do
        if (left == right) then
            left = left+1
        end if
        if (left < (low+high)/2) then
            stack(1,stack_ptr) = left
            stack(2,stack_ptr) = high
            stack_ptr = stack_ptr+1
            high = left-1
        else
            stack(1,stack_ptr) = low
            stack(2,stack_ptr) = left-1
            stack_ptr = stack_ptr+1
            low = left
        end if
    enddo
end subroutine quicksort_nr

end module misc
