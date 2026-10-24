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
