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

static inline void doacross_spin (unsigned long *addr, unsigned long expected,
				  unsigned long cur)
{
  /* FIXME: back off depending on how large expected - cur is.  */
  unsigned n = 0;
  do
    {
      doacross_relax (&n);
      cur = __atomic_load_n (addr, MEMMODEL_RELAXED);
      if (expected < cur)
	return;
    }
  while (1);
}

#ifdef HAVE_ATTRIBUTE_VISIBILITY
# pragma GCC visibility pop
#endif

#endif /* GOMP_DOACROSS_H */
