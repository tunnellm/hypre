/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "schwarz.h"
#include <assert.h>

/* Domain rows reference the full DOF space, not only the local domain size. */
int main(void)
{
   const HYPRE_Int n = 257;
   HYPRE_Int i, maximum;
   HYPRE_Initialize();
   hypre_SchwarzData *data = (hypre_SchwarzData *) hypre_SchwarzCreate();
   hypre_CSRMatrix *domains = hypre_CSRMatrixCreate(n - 1, n, 2 * (n - 1));
   hypre_CSRMatrixInitialize(domains);
   for (i = 0; i < n - 1; i++)
   {
      hypre_CSRMatrixI(domains)[i] = 2 * i;
      hypre_CSRMatrixJ(domains)[2 * i] = i;
      hypre_CSRMatrixJ(domains)[2 * i + 1] = i + 1;
   }
   hypre_CSRMatrixI(domains)[n - 1] = 2 * (n - 1);
   assert(hypre_SchwarzSetSymmetric(data, 1) == 0);
   hypre_SchwarzSetDomainStructure(data, domains);
   assert(hypre_SchwarzGetMaxDomainOverlap(data, &maximum) == 0);
   assert(maximum == 2);
   assert(hypre_SchwarzSetSymmetric(data, 0) != 0);
   assert(data->symmetric == 1);
   HYPRE_ClearAllErrors();
   hypre_SchwarzDestroy(data);
   HYPRE_Finalize();
   return 0;
}
