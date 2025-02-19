module list_mod
  ! module which attempts to implement an easy to use basic linked list in fortran
  ! here each item in the list must be a 3 element vector - change the item as required

  use iso_fortran_env

  implicit none

  type :: item
    real(real64), dimension(3) :: r
    type(item), pointer :: next
  end type

  type :: list
    type(item), pointer :: first, last, current
    integer(int32) :: size, index
  contains
    procedure :: create
    procedure :: append
    procedure :: prepend
    procedure :: delete_index
    procedure :: delete_indices
    procedure :: destroy
    procedure :: to_array
    procedure :: print
    procedure :: reverse
  end type

  contains

    subroutine create(self)
      ! "create" a list -- essentially just sets size attribute to zero in case it's not by default

      class(list) :: self
      
      self%size = 0

    end subroutine

    !********************************************************************************

    subroutine create_from_array(self, veclist)
      ! use an array of vectors to create each element on list

      class(list) :: self
      real(real64), dimension(:, :) :: veclist
      integer :: ivec

      call self%create()

      do ivec = 1, size(veclist, 2)
        call self%append(veclist(:, ivec))
      enddo

    end subroutine

    !********************************************************************************

    subroutine append(self, vec)
      ! add a vector to the end of the list

      class(list) :: self
      real(real64), dimension(3) :: vec
      type(item), pointer :: newitem

      allocate(newitem)
      newitem%r = vec
      if (self%size == 0) then
        self%index = 1
        self%first => newitem
        self%current => newitem
      else
        self%last%next => newitem
      endif
      self%last => newitem
      self%size = self%size + 1

    end subroutine

    !********************************************************************************

    subroutine prepend(self, vec)
      ! add a vector to the start of the list

      class(list) :: self
      real(real64), dimension(3) :: vec
      type(item), pointer :: newitem

      allocate(newitem)
      newitem%r = vec
      if (self%size == 0) then
        self%index = 1
        self%last => newitem
        self%current => newitem
      else
        newitem%next => self%first
      endif
      self%first => newitem
      self%size = self%size + 1

    end subroutine

    !********************************************************************************

    subroutine delete_index(self, index)
      ! delete element from the list at specific index

      class(list) :: self
      integer :: index
      integer :: ilist
      type(item), pointer :: delete

      if (index == 1) then
        delete => self%first
        self%first => self%first%next
      else
        if (index >= self%index) then
          self%current => self%first
          self%index = 1
        endif
        do ilist = 1, index - self%index - 1
          self%current => self%current%next
          self%index = self%index + 1
        enddo
        delete => self%current%next
        if (index /= self%size) self%current%next => self%current%next%next
      endif
      deallocate(delete)
      nullify(delete)
      self%size = self%size - 1

    end subroutine

    !********************************************************************************

    subroutine delete_indices(self, indices)
      ! delete multiple elements from the list given by an array of indicies

      class(list) :: self
      integer, dimension(:) :: indices
      integer :: ilist, iind
      type(item), pointer :: delete

      print*, indices

      do iind = 1, size(indices)
        call self%delete_index(indices(iind))
        indices = indices - 1
      enddo

    end subroutine

    !********************************************************************************

    subroutine destroy(self)
      ! empty the list and delete pointers

      class(list) :: self

      do while (self%size > 0)
        self%current => self%first
        if (self%size > 1) self%first => self%first%next
        deallocate(self%current)
        self%size = self%size - 1
      enddo
      nullify(self%first)
      nullify(self%current)
      nullify(self%last)
      
    end subroutine

    !********************************************************************************

    function to_array(self)
      ! return an array containing each element of list

      class(list) :: self
      real(real64), dimension(:, :), allocatable :: to_array
      integer :: i

      allocate(to_array(3, self%size))

      self%current => self%first
      do i = 1, self%size
        to_array(:, i) = self%current%r
        self%current => self%current%next
      enddo
      self%current => self%first
      self%index = 1

    end function

    !********************************************************************************

    subroutine print(self)
      ! print entire list

      class(list) :: self
      integer :: i

      self%current => self%first
      do i = 1, self%size
        print*, self%current%r
        self%current => self%current%next
      enddo

    end subroutine

    !********************************************************************************

    subroutine reverse(self)
      ! completely reverse the list

      class(list) :: self
      integer :: i

      self%last => self%first
      do i = 3, self%size
        self%current => self%last%next
        self%last%next => self%current%next
        self%current%next => self%first
        self%first => self%current
      enddo
      self%current => self%last%next
      nullify(self%last%next)
      self%current%next => self%first
      self%first => self%current

    end subroutine

end module