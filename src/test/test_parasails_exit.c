/* Sequential ParaSails must fail closed even when MPI_Abort returns.
 * Compile this driver together with distributed_ls/ParaSails/Mem.c so the
 * regression checks the source under test, not an older installed library.
 * This POSIX test uses fork to contain the intentionally fatal calls.
 */
#include "../distributed_ls/ParaSails/Common.h"
#include "../distributed_ls/ParaSails/Mem.h"
#include <sys/wait.h>
#include <unistd.h>

static void unexpected_exit_handler(void)
{
   _Exit(77);
}

static int check_fatal(HYPRE_Int block_count)
{
   int status;
   pid_t child = fork();
   if (child < 0) { return 1; }
   if (child == 0)
   {
      alarm(5);
      atexit(unexpected_exit_handler);
      if (block_count < 0)
      {
         PARASAILS_EXIT;
      }
      else
      {
         Mem *m = MemCreate();
         /* Model an exhausted pool without allocating 2 GiB. The fatal
          * boundary must be checked before allocating or indexing blocks. */
         m->num_blocks = block_count;
         MemAlloc(m, 16);
      }
      _Exit(78);
   }
   if (waitpid(child, &status, 0) != child) { return 1; }
   return !WIFEXITED(status) || WEXITSTATUS(status) != EXIT_FAILURE;
}

int main(void)
{
#ifndef HYPRE_SEQUENTIAL
   fprintf(stderr, "This regression requires a sequential Hypre build.\n");
   return 2;
#else
   Mem *m = MemCreate();
   char *a = MemAlloc(m, 16);
   char *b = MemAlloc(m, 32);
   int failed = (m->num_blocks != 1 || a == b);
   a[0] = 1;
   b[31] = 2;
   failed |= (a[0] != 1 || b[31] != 2);
   MemDestroy(m);
   failed |= check_fatal(-1);
   failed |= check_fatal(MEM_MAXBLOCKS);
   failed |= check_fatal(MEM_MAXBLOCKS + 1);
   if (!failed) { puts("ParaSails allocation and fail-closed checks passed"); }
   return failed;
#endif
}
