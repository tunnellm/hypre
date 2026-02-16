/*
 * Test program to validate FLOP and graph op counting across
 * different AMG configurations. Generates a 2D Laplacian and
 * runs AMG with various smoother/coarsening/interpolation combos,
 * printing setup FLOPs, setup graph ops, and cycle complexity.
 */

#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_ls.h"
#include "HYPRE_IJ_mv.h"
#include "_hypre_IJ_mv.h"
#include "HYPRE_krylov.h"

/* Not in public header */
HYPRE_Int hypre_BoomerAMGGetSetupGraphOps(void *data, HYPRE_Real *setup_graph_ops);

/* Build a 2D 5-point Laplacian on an n x n grid */
static void BuildLaplacian2D(MPI_Comm comm, HYPRE_Int n,
                             HYPRE_IJMatrix *ij_A,
                             HYPRE_IJVector *ij_b,
                             HYPRE_IJVector *ij_x)
{
   HYPRE_Int N = n * n;
   HYPRE_Int myid, num_procs;
   HYPRE_Int ilower, iupper;

   hypre_MPI_Comm_rank(comm, &myid);
   hypre_MPI_Comm_size(comm, &num_procs);

   /* Partition rows */
   HYPRE_Int local_size = N / num_procs;
   HYPRE_Int extra = N - local_size * num_procs;
   ilower = myid * local_size;
   ilower += (myid < extra) ? myid : extra;
   if (myid < extra) { local_size++; }
   iupper = ilower + local_size - 1;

   /* Create matrix */
   HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, ij_A);
   HYPRE_IJMatrixSetObjectType(*ij_A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(*ij_A);

   /* Fill matrix */
   for (HYPRE_Int i = ilower; i <= iupper; i++)
   {
      HYPRE_Int nnz = 0;
      HYPRE_Int cols[5];
      HYPRE_Real vals[5];
      HYPRE_Int row = i;
      HYPRE_Int ix = i % n;
      HYPRE_Int iy = i / n;

      if (iy > 0)     { cols[nnz] = i - n; vals[nnz] = -1.0; nnz++; }
      if (ix > 0)     { cols[nnz] = i - 1; vals[nnz] = -1.0; nnz++; }
      cols[nnz] = i; vals[nnz] = 4.0; nnz++;
      if (ix < n - 1) { cols[nnz] = i + 1; vals[nnz] = -1.0; nnz++; }
      if (iy < n - 1) { cols[nnz] = i + n; vals[nnz] = -1.0; nnz++; }

      HYPRE_IJMatrixSetValues(*ij_A, 1, &nnz, &row, cols, vals);
   }
   HYPRE_IJMatrixAssemble(*ij_A);

   /* Create vectors */
   HYPRE_IJVectorCreate(comm, ilower, iupper, ij_b);
   HYPRE_IJVectorSetObjectType(*ij_b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(*ij_b);

   HYPRE_IJVectorCreate(comm, ilower, iupper, ij_x);
   HYPRE_IJVectorSetObjectType(*ij_x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(*ij_x);

   for (HYPRE_Int i = ilower; i <= iupper; i++)
   {
      HYPRE_Real val = 1.0;
      HYPRE_IJVectorSetValues(*ij_b, 1, &i, &val);
      val = 0.0;
      HYPRE_IJVectorSetValues(*ij_x, 1, &i, &val);
   }
   HYPRE_IJVectorAssemble(*ij_b);
   HYPRE_IJVectorAssemble(*ij_x);
}

typedef struct
{
   const char *name;
   HYPRE_Int   coarsen_type;   /* -1 = default */
   HYPRE_Int   interp_type;    /* -1 = default */
   HYPRE_Int   relax_type;     /* -1 = default (sets all phases) */
   HYPRE_Int   relax_down;     /* -1 = use relax_type; >=0 = pre-smooth type */
   HYPRE_Int   relax_up;       /* -1 = use relax_type; >=0 = post-smooth type */
   HYPRE_Int   smooth_type;    /* -1 = not set */
   HYPRE_Int   smooth_num_levels; /* 0 = not set */
   HYPRE_Int   agg_levels;     /* 0 = default */
   HYPRE_Int   agg_interp;    /* -1 = default (4=multipass), else agg_interp_type */
   HYPRE_Int   sabs;           /* 0 = standard SOC, 1 = sabs */
} TestConfig;

static void RunTest(MPI_Comm comm, HYPRE_Int grid_size, TestConfig *config)
{
   HYPRE_IJMatrix ij_A;
   HYPRE_IJVector ij_b, ij_x;
   HYPRE_ParCSRMatrix A;
   HYPRE_ParVector b, x;
   HYPRE_Solver solver;
   HYPRE_Real setup_flops = 0.0, setup_graph_ops = 0.0;
   HYPRE_Real cycle_op_count;
   HYPRE_Int myid;

   hypre_MPI_Comm_rank(comm, &myid);

   BuildLaplacian2D(comm, grid_size, &ij_A, &ij_b, &ij_x);
   HYPRE_IJMatrixGetObject(ij_A, (void **) &A);
   HYPRE_IJVectorGetObject(ij_b, (void **) &b);
   HYPRE_IJVectorGetObject(ij_x, (void **) &x);

   HYPRE_BoomerAMGCreate(&solver);
   HYPRE_BoomerAMGSetMaxIter(solver, 1);
   HYPRE_BoomerAMGSetTol(solver, 0.0);
   HYPRE_BoomerAMGSetPrintLevel(solver, 0);

   if (config->coarsen_type >= 0)
      HYPRE_BoomerAMGSetCoarsenType(solver, config->coarsen_type);
   if (config->interp_type >= 0)
      HYPRE_BoomerAMGSetInterpType(solver, config->interp_type);
   if (config->relax_down >= 0 && config->relax_up >= 0)
   {
      /* Separate pre/post smoother types */
      HYPRE_BoomerAMGSetCycleRelaxType(solver, config->relax_down, 1);
      HYPRE_BoomerAMGSetCycleRelaxType(solver, config->relax_up, 2);
      HYPRE_BoomerAMGSetCycleRelaxType(solver, 9, 3); /* GE coarse */
   }
   else if (config->relax_type >= 0)
   {
      HYPRE_BoomerAMGSetRelaxType(solver, config->relax_type);
   }
   if (config->smooth_type >= 0)
   {
      HYPRE_BoomerAMGSetSmoothType(solver, config->smooth_type);
      HYPRE_BoomerAMGSetSmoothNumLevels(solver, config->smooth_num_levels);
      HYPRE_BoomerAMGSetSmoothNumSweeps(solver, 1);
   }
   if (config->agg_levels > 0)
      HYPRE_BoomerAMGSetAggNumLevels(solver, config->agg_levels);
   if (config->agg_interp >= 0)
      HYPRE_BoomerAMGSetAggInterpType(solver, config->agg_interp);
   if (config->sabs)
      HYPRE_BoomerAMGSetSabs(solver, 1); /* Use absolute-value SOC */

   HYPRE_BoomerAMGSetup(solver, A, b, x);

   /* Reset x to zero for solve */
   HYPRE_ParVectorSetConstantValues(x, 0.0);
   HYPRE_BoomerAMGSolve(solver, A, b, x);

   /* Get work counts */
   hypre_BoomerAMGGetSetupFlops((void *) solver, &setup_flops);
   hypre_BoomerAMGGetSetupGraphOps((void *) solver, &setup_graph_ops);

   /* Get cycle op count and nnz for complexity */
   {
      hypre_ParAMGData *amg_data = (hypre_ParAMGData *) solver;
      cycle_op_count = hypre_ParAMGDataCycleOpCount(amg_data);
   }

   HYPRE_Real nnz_A = hypre_ParCSRMatrixDNumNonzeros((hypre_ParCSRMatrix *) A);
   HYPRE_BigInt N = hypre_ParCSRMatrixGlobalNumRows((hypre_ParCSRMatrix *) A);

   if (myid == 0)
   {
      HYPRE_Real nnz_per_row = nnz_A / (HYPRE_Real) N;
      hypre_printf("%-35s  N=%6lld  nnz/row=%.1f  "
                   "setup_fma=%.0f (%.1f/nnz)  setup_graph=%.0f (%.1f/nnz)  "
                   "cycle_cmplx=%.4f\n",
                   config->name,
                   (long long) N,
                   nnz_per_row,
                   setup_flops,
                   setup_flops / nnz_A,
                   setup_graph_ops,
                   setup_graph_ops / nnz_A,
                   (nnz_A > 0) ? cycle_op_count / nnz_A : 0.0);
   }

   HYPRE_BoomerAMGDestroy(solver);
   HYPRE_IJMatrixDestroy(ij_A);
   HYPRE_IJVectorDestroy(ij_b);
   HYPRE_IJVectorDestroy(ij_x);
}

int main(int argc, char *argv[])
{
   MPI_Comm comm = MPI_COMM_WORLD;
   HYPRE_Int grid_size = 50; /* 50x50 = 2500 unknowns */

   hypre_MPI_Init(&argc, &argv);
   HYPRE_Init();

   HYPRE_Int myid;
   hypre_MPI_Comm_rank(comm, &myid);

   if (myid == 0)
   {
      hypre_printf("=== Work Counting Validation ===\n");
      hypre_printf("Grid: %d x %d = %d unknowns, 2D 5-pt Laplacian\n\n",
                   grid_size, grid_size, grid_size * grid_size);
      hypre_printf("%-35s  %-8s %-8s  %-30s  %-30s  %s\n",
                   "Config", "N", "nnz/row",
                   "setup_fma (per nnz)", "setup_graph (per nnz)", "cycle_cmplx");
      hypre_printf("----------------------------------------------"
                   "----------------------------------------------"
                   "--------------------------------------\n");
   }

   /* Test configurations */
   TestConfig configs[] = {
      /* name, coarsen, interp, relax, relax_down, relax_up, smooth, sm_levels, agg, agg_interp, sabs */

      /* Baseline: default AMG (HMIS + ext+i interp + hybrid GS) */
      {"Default AMG (HMIS/ext+i/hybGS)", 10, 6, 6, -1, -1, -1, 0, 0, -1, 0},

      /* Coarsening variants */
      {"RS coarsening",                   0, 6, 6, -1, -1, -1, 0, 0, -1, 0},
      {"RS/direct interp",                0, 3, 6, -1, -1, -1, 0, 0, -1, 0},
      {"PMIS coarsening",                 8, 6, 6, -1, -1, -1, 0, 0, -1, 0},
      {"CGC coarsening",                 21, 6, 6, -1, -1, -1, 0, 0, -1, 0},

      /* Interpolation variants */
      {"Classical interp (type 0)",      10, 0, 6, -1, -1, -1, 0, 0, -1, 0},
      {"Direct interp (type 3)",         10, 3, 6, -1, -1, -1, 0, 0, -1, 0},
      {"Multipass interp (type 5)",      10, 5, 6, -1, -1, -1, 0, 0, -1, 0},
      {"Extended+e interp (type 14)",    10, 14, 6, -1, -1, -1, 0, 0, -1, 0},

      /* Relaxation variants */
      {"Jacobi smoother",               10, 6, 0, -1, -1, -1, 0, 0, -1, 0},
      {"Hybrid GS/SOR",                 10, 6, 6, -1, -1, -1, 0, 0, -1, 0},
      {"Symm GS/SOR",                   10, 6, 8, -1, -1, -1, 0, 0, -1, 0},
      {"Fwd/Bwd GS (HMIS/ext+i)",        10, 6, -1, 3, 4, -1, 0, 0, -1, 0},
      {"Fwd/Bwd GS (RS/direct)",         0, 3, -1, 3, 4, -1, 0, 0, -1, 0},
      {"L1 Fwd/Bwd GS",                 10, 6, -1, 13, 14, -1, 0, 0, -1, 0},
      {"L1 Jacobi",                     10, 6, 18, -1, -1, -1, 0, 0, -1, 0},
      {"L1 hybrid GS",                  10, 6, 13, -1, -1, -1, 0, 0, -1, 0},
      {"Chebyshev (order 2)",           10, 6, 16, -1, -1, -1, 0, 0, -1, 0},
      {"CG smoother",                   10, 6, 15, -1, -1, -1, 0, 0, -1, 0},

      /* External smoothers */
      {"ILU(0) smoother",               10, 6, -1, -1, -1, 5, 1, 0, -1, 0},
      {"FSAI smoother",                 10, 6, -1, -1, -1, 4, 1, 0, -1, 0},
      {"Schwarz smoother",              10, 6, -1, -1, -1, 6, 1, 0, -1, 0},

      /* Aggressive coarsening (default: multipass, single-stage) */
      {"Agg coarsening (1 level)",      10, 6, 6, -1, -1, -1, 0, 1, -1, 0},
      /* Two-stage aggressive coarsening (P1*P2 composition) */
      {"Agg 2-stage ext+i (type 1)",    10, 6, 6, -1, -1, -1, 0, 1,  1, 0},
      {"Agg 2-stage ext (type 3)",      10, 6, 6, -1, -1, -1, 0, 1,  3, 0},
      {"Agg 2-stage mod ext+i (type 6)",10, 6, 6, -1, -1, -1, 0, 1,  6, 0},

      /* CR coarsening (type 99) with classical interpolation */
      {"CR coarsening",                  99, 0, 6, -1, -1, -1, 0, 0, -1, 0},

      /* Sabs SOC variant (absolute value strength) */
      {"Sabs SOC (HMIS/ext+i/hybGS)",   10, 6, 6, -1, -1, -1, 0, 0, -1, 1},
      {"Sabs SOC (RS/direct/fwd+bwd)",   0, 3, -1, 3, 4, -1, 0, 0, -1, 1},
   };

   HYPRE_Int num_configs = sizeof(configs) / sizeof(configs[0]);

   for (HYPRE_Int i = 0; i < num_configs; i++)
   {
      RunTest(comm, grid_size, &configs[i]);
   }

   if (myid == 0)
   {
      hypre_printf("\nDone.\n");
   }

   HYPRE_Finalize();
   hypre_MPI_Finalize();
   return 0;
}
