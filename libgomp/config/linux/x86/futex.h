/* Copyright (C) 2005-2023 Free Software Foundation, Inc.
   Contributed by Richard Henderson <rth@redhat.com>.

   This file is part of the GNU Offloading and Multi Processing Library
   (libgomp).

   Libgomp is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   Libgomp is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
   more details.

   Under Section 7 of GPL version 3, you are granted additional
   permissions described in the GCC Runtime Library Exception, version
   3.1, as published by the Free Software Foundation.

   You should have received a copy of the GNU General Public License and
   a copy of the GCC Runtime Library Exception along with this program;
   see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
   <http://www.gnu.org/licenses/>.  */

/* Provide target-specific access to the futex system call.  */

#ifdef __x86_64__
# ifndef SYS_futex
#  define SYS_futex	202
# endif

static inline long
__futex_wait (int *addr, int futex_op, int val)
{
  long res;

  register void *timeout __asm ("r10") = NULL;
  __asm volatile ("syscall"
		  : "=a" (res)
		  : "0" (SYS_futex), "D" (addr), "S" (futex_op),
		    "d" (val), "r" (timeout)
		  : "r11", "rcx", "memory");
  return res;
}

static inline long
__futex_wake (int *addr, int futex_op, int count)
{
  long res;

  __asm volatile ("syscall"
		  : "=a" (res)
		  : "0" (SYS_futex), "D" (addr), "S" (futex_op),
		    "d" (count)
		  : "r11", "rcx", "memory");
  return res;
}
#else
# ifndef SYS_futex
#  define SYS_futex	240
# endif

static inline long
__futex_wait (int *addr, int futex_op, int val)
{
  long res;

  void *timeout = NULL;
  __asm volatile ("int $0x80"
		  : "=a" (res)
		  : "0" (SYS_futex), "b" (addr), "c" (futex_op),
		    "d" (val), "S" (timeout)
		  : "memory");
  return res;
}

static inline long
__futex_wake (int *addr, int futex_op, int count)
{
  long res;

  __asm volatile ("int $0x80"
		  : "=a" (res)
		  : "0" (SYS_futex), "b" (addr), "c" (futex_op),
		    "d" (count)
		  : "memory");
  return res;
}
#endif /* __x86_64__ */

/* Lithe (config/linux/lithe_coop.h): liblithe exports its own futex_wait,
   so libgomp's inline versions take different assembler names.  */
#define futex_wait gomp_futex_wait_fn
#define futex_wake gomp_futex_wake_fn

static inline void
gomp_sys_futex_wait (int *addr, int val)
{
  long err = __futex_wait (addr, gomp_futex_wait, val);

  if (__builtin_expect (err == -ENOSYS, 0))
    {
      gomp_futex_wait &= ~FUTEX_PRIVATE_FLAG;
      gomp_futex_wake &= ~FUTEX_PRIVATE_FLAG;

      __futex_wait (addr, gomp_futex_wait, val);
    }
}

static inline void
gomp_sys_futex_wake (int *addr, int count)
{
  long err = __futex_wake (addr, gomp_futex_wake, count);

  if (__builtin_expect (err == -ENOSYS, 0))
    {
      gomp_futex_wait &= ~FUTEX_PRIVATE_FLAG;
      gomp_futex_wake &= ~FUTEX_PRIVATE_FLAG;

      __futex_wake (addr, gomp_futex_wake, count);
    }
}

#include "../lithe_coop.h"

/* A Lithe context parks on liblithe's futex emulation, so its hart runs
   other contexts meanwhile; anything else (a kernel thread, a context that
   cannot block) takes the kernel futex and is counted, so wakes know
   whether the syscall is needed.  */
static inline void
futex_wait (int *addr, int val)
{
  if (gomp_lithe_can_block ())
    {
      gomp_lithe_futex_wait_ (addr, val);
      return;
    }
  if (__builtin_expect (gomp_lithe_active, 0))
    {
      __atomic_add_fetch (&gomp_lithe_kwaiters, 1, __ATOMIC_SEQ_CST);
      gomp_sys_futex_wait (addr, val);
      __atomic_sub_fetch (&gomp_lithe_kwaiters, 1, __ATOMIC_RELAXED);
      return;
    }
  gomp_sys_futex_wait (addr, val);
}

/* Waiters may be parked contexts and kernel threads at once; wake both.
   The fence orders the caller's store to *ADDR before the kwaiters load,
   against the waiter's increment before its syscall re-reads *ADDR.  */
static inline void
futex_wake (int *addr, int count)
{
  if (__builtin_expect (gomp_lithe_active, 0))
    {
      gomp_lithe_futex_wake_some_ (addr, count);
      __atomic_thread_fence (__ATOMIC_SEQ_CST);
      if (__atomic_load_n (&gomp_lithe_kwaiters, __ATOMIC_RELAXED) == 0)
	return;
    }
  gomp_sys_futex_wake (addr, count);
}

static inline void
cpu_relax (void)
{
  __builtin_ia32_pause ();
}
