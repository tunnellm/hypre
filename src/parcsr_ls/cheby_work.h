/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_CHEBY_WORK_HEADER
#define HYPRE_CHEBY_WORK_HEADER

/* Keep standalone and AMG work accounting identical for the host extension.
 * Matvecs cost nnz in the FMA model; a multiply-add counts as one operation. */
static inline HYPRE_Real
hypre_ChebyApplyWork(HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                     HYPRE_Real nnz, HYPRE_Real n)
{
   if (order <= 4)
   {
      return (HYPRE_Real) order * (nnz + (scale ? 3.0 : 1.0) * n);
   }
   /* First step: 3n (4n scaled); later steps: 4n (5n scaled).
    * The alternative's final Richardson step saves one multiply per row. */
   return (HYPRE_Real) order * nnz +
          ((scale ? 5.0 : 4.0) * order - 1.0 - (variant == 1)) * n;
}

static inline HYPRE_Real
hypre_ChebyRecurrenceSetupWork(HYPRE_Int order, HYPRE_Int variant)
{
   return order > 4 ? 4.0 * order - (variant == 1 ? 3.0 : 0.0) : 0.0;
}
#endif
