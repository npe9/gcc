/* Copyright (C) 2015-2023 Free Software Foundation, Inc.
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

/* This is a Linux specific implementation of doacross spinning.  */

#ifndef GOMP_DOACROSS_H
#define GOMP_DOACROSS_H 1

#include "libgomp.h"
#include <errno.h>
#include "wait.h"
#include <limits.h>

#ifdef HAVE_ATTRIBUTE_VISIBILITY
# pragma GCC visibility push(hidden)
#endif

/* One iteration of a doacross wait (doacross_spin and the GOMP_doacross_wait
   loops in ordered.c).  Lithe: the wait is a pure spin on another iteration's
   post, and post never wakes anyone, so with more team threads than harts the
   waiters holding the harts spin on iterations owned by hartless threads and
   never progress.  MEASURED 2026-10-24 (.build_logs/spin_census/doacross,
   ordered(1) chain, 4 harts): 8 and 16 threads hung 3/3 (30 s timeout); stock
   libgomp needs 28.6 s at 8 threads on 4 cores.  GOMP_LITHE_DOACROSS_YIELD
   (default 1; =0 is the old pure spin): every 1024 iterations, if a runnable
   peer has no hart, yield the hart to it (a yield, not a park: nothing would
   wake a parked waiter).  n=5 interleaved: off 8 and 16 threads hang 10/10;
   on 8 threads 35-70 ms, 16 threads 83-163 ms; 4 threads (1:1) unchanged
   12 ms; checksums identical.  */
static inline void
doacross_relax (unsigned *n)
{
  if (__builtin_expect ((++*n & 1023) == 0, 0)
      && gomp_lithe_should_stop_spinning ()
      && gomp_lithe_doacross_yield_on ())
    lithe_context_yield ();
  else
    cpu_relax ();
}

/* GOMP_LITHE_DOACROSS_BLOCK (default 1 since 2026-10-24; supersedes the yield
   above, which is the =0 path): a doacross wait BLOCKS on a Lithe futex and
   the post wakes it.  The
   futex words are a fixed table of 64 {seq, waiters} slots (team.c), indexed
   by the awaited array entry's cache line, so the set of futex addresses is
   bounded: liblithe's futex keeps every address it has seen on one linear
   list that never shrinks (src/lithe/src/futex.c get_futex_list), and a word
   per entry made schedule(dynamic,1) chains quadratic (get_futex_list 62 % of
   cycles, 6x slower than the yield; .build_logs/despin/openmp/doacross).
   Waiter: waiters++, fence, read seq, re-check the counters, futex_wait
   (&seq, s) while still unsatisfied; waiters-- on the way out.  Poster: after
   its counter store, fence; if the slot has waiters, seq++ and wake all (one
   relaxed load when nobody waits; a slot collision is a spurious wake).
   Only on a context that may block (gomp_lithe_can_block); everyone else
   keeps the relax path.  GOMP_LITHE_DOACROSS_SPIN=N (default 4000): up to N
   pause iterations before the first block, cut short as soon as a runnable
   peer has no hart, so it only spins while every runnable context has a hart.
   MEASURED 2026-10-24 (.build_logs/despin/openmp/doacross/ab2.csv, 4 harts,
   n=5 interleaved, ms; chain = ordered(1) static,1 / ull / 2-D / dynamic,1):
   4 threads (1:1)   yield 13.5 12.5  2.1 12.4  block+spin 12.7 13.2 2.6 12.2
   8 threads         yield 38.7 45.4  3.5 34.2  block+spin 14.4 14.6 2.9 15.2
   16 threads        yield 125  124   4.9 48.8  block+spin 15.4 15.1 2.9 16.9
   (=0 spin: 1:1 14.5-15.0, the park/wake on the critical chain, so the spin
   is kept; a spin that ignores demand, 10000 pauses, made 8-16 threads
   220 ms).  Checksums identical in every arm, no failures.  The 2-D 1:1 case
   +0.5 ms is inside its range (2.1-3.6 vs 2.0-15).  */
struct gomp_lithe_dx_slot
{
  int seq;
  int waiters;
  char pad[56];
};
extern struct gomp_lithe_dx_slot gomp_lithe_dx_slots[64];
extern int gomp_lithe_doacross_block_mode;	/* -1 unread, 0 off, 1 on */
extern unsigned gomp_lithe_doacross_spin;
extern int gomp_lithe_doacross_block_init (void);

static inline int
gomp_lithe_doacross_block_on (void)
{
  int m = __atomic_load_n (&gomp_lithe_doacross_block_mode, MEMMODEL_RELAXED);
  if (__builtin_expect (m < 0, 0))
    m = gomp_lithe_doacross_block_init ();
  return m;
}

struct doacross_lithe_wait
{
  int armed;	/* waiters incremented, s valid */
  int s;	/* seq read after arming */
};

static inline int *
doacross_lithe_word (struct gomp_doacross_work_share *d, void *elt)
{
  (void) d;
  uintptr_t h = (uintptr_t) elt >> 6;
  return &gomp_lithe_dx_slots[(h ^ (h >> 6)) & 63].seq;
}

static inline void
doacross_lithe_post (struct gomp_doacross_work_share *d, void *elt)
{
  if (__builtin_expect (gomp_lithe_active, 0)
      && gomp_lithe_doacross_block_on ())
    {
      int *w = doacross_lithe_word (d, elt);
      __atomic_thread_fence (MEMMODEL_SEQ_CST);
      if (__atomic_load_n (&w[1], MEMMODEL_RELAXED) != 0)
	{
	  __atomic_add_fetch (&w[0], 1, MEMMODEL_RELEASE);
	  futex_wake (&w[0], INT_MAX);
	}
    }
}

/* Called with the condition just found false; the caller re-checks it after
   every return.  */
static inline void
doacross_relax2 (struct doacross_lithe_wait *lw,
		 struct gomp_doacross_work_share *d, void *elt, unsigned *n)
{
  if (__builtin_expect (gomp_lithe_active, 0)
      && gomp_lithe_doacross_block_on () && gomp_lithe_can_block ())
    {
      int *w = doacross_lithe_word (d, elt);
      if (*n < gomp_lithe_doacross_spin)
	{
	  /* Demand-aware: the spin ends as soon as a runnable peer has no
	     hart (it may be the poster).  */
	  if ((*n & 63) == 0 && gomp_lithe_should_stop_spinning ())
	    *n = gomp_lithe_doacross_spin;
	  else
	    {
	      ++*n;
	      cpu_relax ();
	      return;
	    }
	}
      if (!lw->armed)
	{
	  __atomic_add_fetch (&w[1], 1, MEMMODEL_SEQ_CST);
	  __atomic_thread_fence (MEMMODEL_SEQ_CST);
	  lw->s = __atomic_load_n (&w[0], MEMMODEL_ACQUIRE);
	  lw->armed = 1;
	  return;
	}
      futex_wait (&w[0], lw->s);
      lw->s = __atomic_load_n (&w[0], MEMMODEL_ACQUIRE);
      return;
    }
  doacross_relax (n);
}

static inline void
doacross_lithe_wait_end (struct doacross_lithe_wait *lw,
			 struct gomp_doacross_work_share *d, void *elt)
{
  if (lw->armed)
    __atomic_sub_fetch (&doacross_lithe_word (d, elt)[1], 1,
			MEMMODEL_RELAXED);
}

static inline void doacross_spin (struct gomp_doacross_work_share *d,
				  unsigned long *addr, unsigned long expected,
				  unsigned long cur)
{
  /* FIXME: back off depending on how large expected - cur is.  */
  unsigned n = 0;
  struct doacross_lithe_wait lw = { 0, 0 };
  do
    {
      doacross_relax2 (&lw, d, addr, &n);
      cur = __atomic_load_n (addr, MEMMODEL_RELAXED);
      if (expected < cur)
	{
	  doacross_lithe_wait_end (&lw, d, addr);
	  return;
	}
    }
  while (1);
}

#ifdef HAVE_ATTRIBUTE_VISIBILITY
# pragma GCC visibility pop
#endif

#endif /* GOMP_DOACROSS_H */
