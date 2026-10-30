/* Copyright (C) 2008-2023 Free Software Foundation, Inc.
   Contributed by Jakub Jelinek <jakub@redhat.com>.

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

/* This is a Linux specific implementation of a mutex synchronization
   mechanism for libgomp.  This type is private to the library.  This
   implementation uses atomic instructions and the futex syscall.  */

#ifndef GOMP_WAIT_H
#define GOMP_WAIT_H 1

#include "libgomp.h"
#include <errno.h>
#include <time.h>

#define FUTEX_WAIT	0
#define FUTEX_WAKE	1
#define FUTEX_PRIVATE_FLAG	128

#ifdef HAVE_ATTRIBUTE_VISIBILITY
# pragma GCC visibility push(hidden)
#endif

extern int gomp_futex_wait, gomp_futex_wake;

#include <futex.h>

/* GOMP_LITHE_SPIN_YIELD: the spin as Lithe yields (lithe_coop.h).  Returns 1
   if the word did not change within the budget (caller parks).  */
static int __attribute__ ((noinline))
do_spin_lithe_yield (int *addr, int val, unsigned long us)
{
  struct timespec t0, t;
  clock_gettime (CLOCK_MONOTONIC, &t0);
  unsigned long long budget = us * 1000ULL;
  for (unsigned i = 0; ; i++)
    {
      if (__atomic_load_n (addr, MEMMODEL_RELAXED) != val)
	return 0;
      lithe_context_yield ();
      if ((i & 15) == 15)
	{
	  clock_gettime (CLOCK_MONOTONIC, &t);
	  if ((unsigned long long) ((t.tv_sec - t0.tv_sec) * 1000000000LL
				    + (t.tv_nsec - t0.tv_nsec)) >= budget)
	    return 1;
	}
    }
}

/* GOMP_LITHE_NOSPIN (team.c; default = on iff GOMP_LITHE_OWN_SCHED=1): on a
   Lithe context that may block, no spin at all -- every futex-word wait (team
   barrier, pool dock, mutex, sem, ptrlock, ordered) parks on the Lithe futex
   at once, and the hart goes back to the scheduler.  Supersedes
   GOMP_LITHE_SPIN_YIELD when on.  Kernel threads and contexts that cannot
   block keep GOMP_SPINCOUNT.
   MEASURED 2026-10-24 (cores 32-35, 4 harts, n=5 interleaved,
   .build_logs/despin/openmp/{ab_libgomp.csv,ab_misc.csv,climml/ns1}):
   flat (the default composition) it LOSES where the spin bridges short gaps:
   region cost 4 threads 1.77 -> 3.86 us, 8 threads 7.01 -> 7.25; contended
   omp_lock 8 threads (1000/1000) 0.114 -> 0.151 s; OpenMP-alone in the TBB/HPX
   mixes +3 %; neutral on the X+OpenMP mixes (TBB 0.153/0.153, HPX 0.166/0.160,
   PaRSEC 0.268/0.261, StarPU 0.144/0.145 s), nested, tasks, climml l1f/l2f
   (8.98/9.12, 9.01/9.07 ms/step); better for ordered at 8/16 threads
   (6.0 -> 4.8, 6.5 -> 5.3 ms).  The existing spin already stops when a
   runnable peer has no hart (gomp_lithe_should_stop_spinning), which is the
   demand-aware part.  What that check cannot see is a scheduler boundary:
   with GOMP_LITHE_OWN_SCHED=1 the dock spin holds the child scheduler's harts
   against the master's sched_exit drain -- climml l1s 228.5 -> 19.0, l2s
   228.7 -> 19.0 ms/step (= GOMP_SPINCOUNT=0, 18.7).  Hence the default.  */
extern int gomp_lithe_nospin_mode;
extern int gomp_lithe_nospin_init (void);

static inline int do_spin (int *addr, int val)
{
  unsigned long long i, count = gomp_spin_count_var;

  if (__builtin_expect (gomp_lithe_active, 0))
    {
      int m = __atomic_load_n (&gomp_lithe_nospin_mode, MEMMODEL_RELAXED);
      if (__builtin_expect (m < 0, 0))
	m = gomp_lithe_nospin_init ();
      if (m && lithe_can_block ())
	return __atomic_load_n (addr, MEMMODEL_RELAXED) == val;
    }

  if (__builtin_expect (gomp_lithe_active, 0) && gomp_lithe_spin_yield_us ()
      && lithe_context_yield && lithe_can_block && lithe_can_block ())
    return do_spin_lithe_yield (addr, val, gomp_lithe_spin_yield_us ());

  if (__builtin_expect (__atomic_load_n (&gomp_managed_threads,
                                         MEMMODEL_RELAXED)
                        > gomp_available_cpus, 0))
    count = gomp_throttled_spin_count_var;
  int dy;
  if (__builtin_expect (gomp_lithe_active, 0)
      && (dy = gomp_lithe_spin_demand_yield ()) != 0
      && lithe_context_yield && lithe_can_block && lithe_can_block ()
      && (gomp_lithe_spin_demand_min_harts () == 0 || !parlib_vcore_limit
	  || parlib_vcore_limit () >= gomp_lithe_spin_demand_min_harts ())
      && !(gomp_lithe_spin_demand_skip_foreign ()
	   && gomp_lithe_team_foreign_master ()))
    {
      /* GOMP_LITHE_SPIN_DEMAND_YIELD (lithe_coop.h): yield the hart to a
	 hartless runnable peer, keep spinning otherwise.  */
      for (i = 0; i < count; i++)
	if (__builtin_expect (__atomic_load_n (addr, MEMMODEL_RELAXED) != val, 0))
	  return 0;
	else if (__builtin_expect ((i & (dy - 1)) == 0, 0)
		 && (gomp_lithe_spin_demand_discount ()
		     ? lithe_fork_join_current_hartless_runnable () >
		       __atomic_load_n (&gomp_lithe_demand_yielders,
					MEMMODEL_RELAXED)
		     : gomp_lithe_should_stop_spinning ()))
	  {
	    /* demand: spin as yields for a bounded time, then park; unbounded
	       yielding let spinners trade the hart (LAMMPS over -13 %) */
	    unsigned long us = gomp_lithe_spin_demand_yield_us ();
	    long pa = gomp_lithe_spin_demand_park_at ();
	    if (us == (unsigned long) -1
		|| (pa && lithe_fork_join_current_hartless_runnable () >= pa)
		|| gomp_lithe_demand_elsewhere ())
	      return 1;
	    if (us)
	      {
		int r;
		__atomic_add_fetch (&gomp_lithe_demand_yielders, 1,
				    MEMMODEL_RELAXED);
		r = do_spin_lithe_yield (addr, val, us);
		__atomic_sub_fetch (&gomp_lithe_demand_yielders, 1,
				    MEMMODEL_RELAXED);
		return r;
	      }
	    lithe_context_yield ();
	  }
	else
	  cpu_relax ();
      return 1;
    }
  for (i = 0; i < count; i++)
    if (__builtin_expect (__atomic_load_n (addr, MEMMODEL_RELAXED) != val, 0))
      return 0;
    else if (__builtin_expect ((i & 1023) == 0, 0)
	     && gomp_lithe_should_stop_spinning ())
      /* Lithe: a peer is runnable with no hart; park now instead of spinning
	 on the hart it needs.  */
      return 1;
    else
      cpu_relax ();
  return 1;
}

static inline void do_wait (int *addr, int val)
{
  /* Lithe trace: the whole wait as a phase slice on this context's track.  */
  int tr = gomp_lithe_trace_wait ();
  if (tr)
    lithe_sched_trace_phase_begin (tr);
  if (do_spin (addr, val))
    futex_wait (addr, val);
  if (tr)
    lithe_sched_trace_phase_end ();
}

#ifdef HAVE_ATTRIBUTE_VISIBILITY
# pragma GCC visibility pop
#endif

#endif /* GOMP_WAIT_H */
