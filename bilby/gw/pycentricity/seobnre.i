/* file : seobnre.i */

/* name of module to use*/
%module seobnre

%{
    /* Every thing in this file is being copied in
     wrapper file. We include the C header file necessary
     to compile the interface */
     #define SWIG_FILE_WITH_INIT
     #include <gsl/gsl_vector.h>
     #include <gsl/gsl_multiroots.h>
     #include <gsl/gsl_integration.h>
     #include <gsl/gsl_sf_gamma.h>
     #include <gsl/gsl_matrix.h>
     #include <gsl/gsl_blas.h>
     #include <gsl/gsl_deriv.h>
     #include <gsl/gsl_errno.h>
     #include <gsl/gsl_linalg.h>
     #include <gsl/gsl_interp.h>
     #include <gsl/gsl_spline.h>
     #include "Panyidatatypes.h"
     #include "Panyi_elip.h"
     #include "seobnre.h"

%}

/* if we want to interface all functions then we can simply
   include header file like this -
*/
%include "seobnre.h"