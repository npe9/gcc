/* Lithe co-scheduling for libgomp (lithe-docker, 2026-10-18).

   libgomp's worker threads become Lithe fork-join contexts (team.c) and its
   futex waits park the calling context instead of its hart (x86/futex.h), so
   an OpenMP team shares harts cooperatively with whatever else runs under
   Lithe -- MPI rank contexts, other teams, GPU runtime progress contexts.

   Every Lithe symbol is a weak reference: a process without liblithe gets
   stock libgomp.  GOMP_LITHE=0 also forces stock behaviour.  The switch is
   made once, at the first team start that finds Lithe present; from then on
   gomp_lithe_active is 1 and futex wakes reach both Lithe-parked contexts
   and kernel-futex waiters.  */

#ifndef GOMP_LITHE_COOP_H
#define GOMP_LITHE_COOP_H 1

#include <stddef.h>

#ifdef HAVE_ATTRIBUTE_VISIBILITY
# pragma GCC visibility push(default)
#endif

typedef struct lithe_sched lithe_sched_t;
typedef struct lithe_fork_join_sched lithe_fork_join_sched_t;
typedef struct lithe_fork_join_context lithe_fork_join_context_t;

extern lithe_sched_t *lithe_sched_current (void) __attribute__ ((weak));
extern int parlib_domain_contended (void) __attribute__ ((weak)); /* parlib */
extern int parlib_cpu_contended (void) __attribute__ ((weak)); /* parlib */
extern int parlib_vcore_limit (void) __attribute__ ((weak)); /* parlib */
extern int lithe_can_block (void) __attribute__ ((weak));
extern void lithe_context_yield (void) __attribute__ ((weak));
extern lithe_fork_join_sched_t *lithe_ensure_root_fork_join_sched (void)
  __attribute__ ((weak));
extern lithe_fork_join_context_t *
lithe_fork_join_context_create (lithe_fork_join_sched_t *, size_t,
				void (*) (void *), void *)
  __attribute__ ((weak));
extern void lithe_fork_join_context_set_daemon (lithe_fork_join_context_t *)
  __attribute__ ((weak));
extern long lithe_fork_join_current_hartless_runnable (void)
  __attribute__ ((weak));
/* GOMP_LITHE_OWN_SCHED: this runtime instance's own fork-join scheduler.  */
extern lithe_fork_join_sched_t *lithe_fork_join_sched_create (void)
  __attribute__ ((weak));
extern void lithe_sched_enter (lithe_sched_t *) __attribute__ ((weak));
extern void lithe_sched_exit (void) __attribute__ ((weak));
/* Only its address matters: a fork-join scheduler's first member (funcs)
   points at it.  */
extern const char lithe_fork_join_sched_funcs[] __attribute__ ((weak));
/* Lithe's Perfetto trace (sched_trace.h; no-ops unless LITHE_SCHED_TRACE=1).  */
extern int lithe_sched_trace_enabled (void) __attribute__ ((weak));
extern void lithe_sched_trace_label_self (const char *) __attribute__ ((weak));
extern int lithe_sched_trace_intern (const char *) __attribute__ ((weak));
extern void lithe_sched_trace_phase_begin (int) __attribute__ ((weak));
extern void lithe_sched_trace_phase_end (void) __attribute__ ((weak));
/* liblithe's futex emulation (src/lithe/src/futex.c): one wait queue per
   address, the value re-checked under its lock inside lithe_context_block.
   The names collide with libgomp's own futex_wait/futex_wake, which
   x86/futex.h renames for that reason.  */
extern int gomp_lithe_futex_wait_ (int *, int)
  __asm__ ("futex_wait") __attribute__ ((weak));
extern int gomp_lithe_futex_wake_some_ (int *, int)
  __asm__ ("futex_wake_some") __attribute__ ((weak));

#ifdef HAVE_ATTRIBUTE_VISIBILITY
# pragma GCC visibility pop
#endif

#ifdef HAVE_ATTRIBUTE_VISIBILITY
# pragma GCC visibility push(hidden)
#endif

/* 1 once libgomp has started using Lithe in this process.  */
extern int gomp_lithe_active;
/* Threads (not Lithe contexts) currently in a kernel futex wait, so a wake
   can skip the syscall when there are none.  */
extern int gomp_lithe_kwaiters;

/* Decide (once) whether to use Lithe and bootstrap the root scheduler if the
   caller is not inside Lithe yet.  Returns gomp_lithe_active.  */
extern int gomp_lithe_init (void);
/* The fork-join scheduler new workers are created in: the caller's own when
   it is a fork-join scheduler, else the root.  */
extern lithe_fork_join_sched_t *gomp_lithe_worker_sched (void);

#ifdef HAVE_ATTRIBUTE_VISIBILITY
# pragma GCC visibility pop
#endif

/* The caller is a Lithe context that may park.  False before Lithe is in
   use, on the base scheduler, and in constructors / _dl_fini.  */
static inline int
gomp_lithe_can_block (void)
{
  return __builtin_expect (gomp_lithe_active, 0) && lithe_can_block ();
}

/* A spinning waiter holds its hart; give it up when a runnable peer has no
   hart to run on (the thread it waits for may be that peer).  */
static inline int
gomp_lithe_should_stop_spinning (void)
{
  return __builtin_expect (gomp_lithe_active, 0)
	 && lithe_fork_join_current_hartless_runnable
	 && lithe_can_block ()
	 && lithe_fork_join_current_hartless_runnable () > 0;
}

/* GOMP_LITHE_SPIN_YIELD=1 (default 0 until measured): a spinning waiter's
   spin IS Lithe yields -- each iteration checks the word and then
   lithe_context_yield (), which always enters the fork-join scheduler (no fast
   path), so whoever is runnable gets the hart at once instead of after the
   1024-iteration stop-spinning check.  The spin is bounded in time
   (GOMP_LITHE_SPIN_YIELD_US, default 100), then the waiter parks on the Lithe
   futex as before.  gomp_lithe_spin_yield_us () returns the budget in us, 0
   when off.
   MEASURED 2026-10-24 (climml, 4 harts, ms/step, n=5; .build_logs/mphys/
   climml/spinyield_*): needs liblithe LITHE_FJS_YIELD_REQUEST=1.  Without it
   a yield never requests a hart, a hart that returned to parlib never comes
   back, and 4 contexts run on 3 harts: 16.5 vs 9.4 (5/5).  With it: one
   instance 9.49 vs 9.40 park.  Two instances (8 contexts on 4 harts): the
   idle instance's yield-spinners are runnable and round-robin with the active
   team -- budget 5000 us 10.5-11.1, 2000 10.0-10.2, 500 9.6, <= 100 9.4-9.5 =
   park.  Parity, not a gain, on this workload; default off pending the
   regression set.  */
extern unsigned long gomp_lithe_spin_yield_us (void);

/* GOMP_LITHE_SPIN_DEMAND_YIELD=1 (default 0 until measured): the spin stays a
   spin while nobody needs the hart, and yields it -- instead of parking -- when
   a runnable peer has none (checked every 64 iterations, not 1024).  Neither
   fixed mode fits both cases (2026-10-30, 4 harts, region cost us): plain spin
   1.83 at 4 threads / 6.91 at 8; spin-as-yields 2.75-3.09 / 4.72 (a yield with
   nobody waiting is a scheduler round trip).  GROMACS 2 ranks x 4 threads on 4
   CPUs: spin 148 ns/day, yields 167, park-at-once 163; the stop-spinning park
   costs a block + wake per gap (.build_logs/gmx_gompl_gap).  */
extern int gomp_lithe_spin_demand_yield (void);
/* GOMP_LITHE_SPIN_DEMAND_YIELD_US (default 20): once demand is seen, the rest
   of the wait is GOMP_LITHE_SPIN_YIELD-style yields for this many us, then
   park; 0 = keep yielding for the whole spin count; -1 = park at once.  */
extern unsigned long gomp_lithe_spin_demand_yield_us (void);
/* GOMP_LITHE_SPIN_DEMAND_DISCOUNT=1 (A/B, default 0): demand = more hartless
   runnable contexts than waiters now in the demand-yield phase.  A waiter that
   yields is itself runnable without a hart, so without the discount one
   yielder makes every other spinner see demand and they round-robin among
   themselves (nested 4x4 on 4 harts +14 %).  */
extern int gomp_lithe_spin_demand_discount (void);
extern int gomp_lithe_demand_yielders;
/* MEASURED 2026-10-30 (4 harts, n=10, .build_logs/gmx_gompl_gap/micro_dy4.csv):
   the discount is WRONG -- region 8 threads 0.724 -> 1.339 of spin (yield-
   spinners trading the hart is what helps flat oversubscription), nested 4x4
   unchanged 1.11 -> 1.15.  Kept only for the record; do not enable.

   GOMP_LITHE_SPIN_DEMAND_PARK_AT=N (A/B, default 0 = off): with N or more
   hartless runnable contexts, park instead of yield-spinning (deep
   oversubscription: nested 4x4 parks better, flat 2x yields better).  */
extern long gomp_lithe_spin_demand_park_at (void);
/* GOMP_LITHE_SPIN_DEMAND_XPROC=1 (A/B, default 0; bit 0 needs PARLIB_FORK_DOMAIN=1):
   when demand is seen and the fork domain holds more harts than CPUs (a sibling
   process -- a forked DataLoader worker -- would run if we parked), park instead
   of yield.  The yield hands the hart to our own process; the park hands the CPU
   to the kernel, where the sibling can take it (GAPS B85/B87).  */
extern int gomp_lithe_spin_demand_xproc (void);
/* GOMP_LITHE_SPIN_DEMAND_YIELD_MIN_HARTS=N (A/B, default 0 = any): use the
   demand-yield path only in a process admitted N or more harts.  B87: a forked
   DataLoader worker (1 hart, a 4-thread team) under demand-yield spent 45 % of
   all CPU cycling its idle waiters through the scheduler -- on one hart a yield
   can only take time from the thread that has the work, a park lets it run.  */
extern int gomp_lithe_spin_demand_min_harts (void);
/* =2 (bit 1): park when this hart's CPU is shared with another task right now
   (parlib_cpu_contended: the hart's OS thread was involuntarily preempted in
   the last ~1 ms) -- sees DataLoader workers, which are not Lithe processes
   and so are outside the fork domain.  =3: either.  */
static inline int
gomp_lithe_demand_elsewhere (void)
{
  int x = gomp_lithe_spin_demand_xproc ();
  return x
	 && (((x & 1) && parlib_domain_contended && parlib_domain_contended ())
	     || ((x & 2) && parlib_cpu_contended && parlib_cpu_contended ()));
}
extern int gomp_lithe_doacross_yield_on (void);

/* Trace phase id for a futex-word wait (spin, then park): barriers, locks, team
   start.  0 when tracing is off; computed once.  */
extern int gomp_lithe_trace_wait_id;
static inline int
gomp_lithe_trace_wait (void)
{
  int id = __atomic_load_n (&gomp_lithe_trace_wait_id, MEMMODEL_RELAXED);
  if (__builtin_expect (id < 0, 0))
    {
      id = (lithe_sched_trace_enabled && lithe_sched_trace_enabled ()
	    && lithe_sched_trace_intern)
	   ? lithe_sched_trace_intern ("omp: spin/park wait") : 0;
      __atomic_store_n (&gomp_lithe_trace_wait_id, id, MEMMODEL_RELAXED);
    }
  return id;
}

#endif /* GOMP_LITHE_COOP_H */
