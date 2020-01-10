// $Id: Panyi_elip.cpp,v 1.1.1.1 2017/09/25 02:35:55 zjcao Exp $

#include "Panyi_elip.h"
#include "Panyicomm_elip.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include "PNwaveformPRD544813.h"

#ifdef newc
#include <ccomplex>
#include <cstdlib>
#include <cmath>
#else
#include <complex.h>
#include <stdlib.h>
#include <math.h>
#endif

#define PI M_PI

const LALUnit lalStrainUnit        = {{ 0, 0, 0, 0, 0, 1, 0}, { 0, 0, 0, 0, 0, 0, 0} };	/**< Strain [1] */

int XLALSimInspiralChooseTDWaveform(
    REAL8TimeSeries **hplus,                    /**< +-polarization waveform */
    REAL8TimeSeries **hcross,                   /**< x-polarization waveform */
    REAL8 phiRef,                               /**< reference orbital phase (rad) */
    REAL8 deltaT,                               /**< sampling interval (s) */
    REAL8 m1,                                   /**< mass of companion 1 (kg) */
    REAL8 m2,                                   /**< mass of companion 2 (kg) */
    REAL8 S1x,                                  /**< x-component of the dimensionless spin of object 1 */
    REAL8 S1y,                                  /**< y-component of the dimensionless spin of object 1 */
    REAL8 S1z,                                  /**< z-component of the dimensionless spin of object 1 */
    REAL8 S2x,                                  /**< x-component of the dimensionless spin of object 2 */
    REAL8 S2y,                                  /**< y-component of the dimensionless spin of object 2 */
    REAL8 S2z,                                  /**< z-component of the dimensionless spin of object 2 */
    REAL8 f_min,                                /**< starting GW frequency (Hz) */
    REAL8 e0,                                   /**< eccentricity at starting GW frequency (Hz) */
    REAL8 r,                                    /**< distance of source (m) */
    REAL8 i                                    /**< inclination of source (rad) */
    )
{
 
    int ret;
    /* N.B. the quadrupole of a spinning compact body labeled by A is 
     * Q_A = - quadparam_A chi_A^2 m_A^3 (see gr-qc/9709032)
     * where quadparam = 1 for BH ~= 4-8 for NS.
     * This affects the quadrupole-monopole interaction.
     * For now, hardcode quadparam1,2 = 1.
     * Will later add ability to set via LALSimInspiralTestGRParam
     */
    REAL8 v0 = 1., quadparam1 = 1., quadparam2 = 1.;

    /* General sanity checks that will abort */
    /*
     * If non-GR approximants are added, change the below to
     * if( nonGRparams && approximant != nonGR1 && approximant != nonGR2 )
     */
 
    /* General sanity check the input parameters - only give warnings! */
    if( deltaT > 1. )
        printf("XLAL Warning - : Large value of deltaT = %e requested.\nPerhaps sample rate and time step size were swapped?\n", deltaT);
    if( deltaT < 1./16385. )
        printf("XLAL Warning - : Small value of deltaT = %e requested.\nCheck for errors, this could create very large time series.\n", deltaT);
    if( m1 < 0.09 * LAL_MSUN_SI )
        printf("XLAL Warning - : Small value of m1 = %e (kg) = %e (Msun) requested.\nPerhaps you have a unit conversion error?\n", m1, m1/LAL_MSUN_SI);
    if( m2 < 0.09 * LAL_MSUN_SI )
        printf("XLAL Warning - : Small value of m2 = %e (kg) = %e (Msun) requested.\nPerhaps you have a unit conversion error?\n", m2, m2/LAL_MSUN_SI);
    if( m1 + m2 > 1000. * LAL_MSUN_SI )
        printf("XLAL Warning - : Large value of total mass m1+m2 = %e (kg) = %e (Msun) requested.\nSignal not likely to be in band of ground-based detectors.\n", m1+m2, (m1+m2)/LAL_MSUN_SI);
    if( S1x*S1x + S1y*S1y + S1z*S1z > 1.000001 )
        printf("XLAL Warning - : S1 = (%e,%e,%e) with norm > 1 requested.\nAre you sure you want to violate the Kerr bound?\n", S1x, S1y, S1z);
    if( S2x*S2x + S2y*S2y + S2z*S2z > 1.000001 )
        printf("XLAL Warning - : S2 = (%e,%e,%e) with norm > 1 requested.\nAre you sure you want to violate the Kerr bound?\n", S2x, S2y, S2z);
    if( f_min < 1. )
        printf("XLAL Warning - : Small value of fmin = %e requested.\nCheck for errors, this could create a very long waveform.\n",f_min);
    if( f_min > 40.000001 )
        printf("XLAL Warning - : Large value of fmin = %e requested.\nCheck for errors, the signal will start in band.\n", f_min);
    if( e0 < 0 || e0 > 1 )
        printf("XLAL Warning - : unphysical e0 = %e.\n", e0);

   

   /* Call the waveform driver routine */
   ret = XLALSimIMRSpinAlignedEOBWaveform(hplus, hcross, phiRef, 
                    deltaT, m1, m2, f_min, e0, r, i, S1z, S2z); // in current file

    return ret;
}

/**��Ҫ�ļ��������
 * This function generates spin-aligned SEOBNRv1 waveforms h+ and hx.  
 * Currently, only the h22 harmonic is available.
 * STEP 0) Prepare parameters, including pre-computed coefficients 
 *         for EOB Hamiltonian, flux and waveform
 * STEP 1) Solve for initial conditions
 * STEP 2) Evolve EOB trajectory until reaching the peak of orbital frequency
 * STEP 3) Step back in time by tStepBack and volve EOB trajectory again 
 *         using high sampling rate, stop at 0.3M out of the "EOB horizon".
 * STEP 4) Locate the peak of orbital frequency for NQC and QNM calculations
 * STEP 5) Calculate NQC correction using hi-sampling data
 * STEP 6) Calculate QNM excitation coefficients using hi-sampling data
 * STEP 7) Generate full inspiral waveform using desired sampling frequency
 * STEP 8) Generate full IMR modes -- attaching ringdown to inspiral
 * STEP 9) Generate full IMR hp and hx waveforms
 */
int XLALSimIMRSpinAlignedEOBWaveform(
        REAL8TimeSeries **hplus,     /**<< OUTPUT, +-polarization waveform */
        REAL8TimeSeries **hcross,    /**<< OUTPUT, x-polarization waveform */
        const REAL8     phiC,        /**<< coalescence orbital phase (rad) */ 
        REAL8           deltaT,      /**<< sampling time step */
        const REAL8     m1SI,        /**<< mass-1 in SI unit */ 
        const REAL8     m2SI,        /**<< mass-2 in SI unit */
        const REAL8     fMin,        /**<< starting frequency (Hz) */
        const REAL8     e0,          /**<< eccentricity at starting GW frequency (Hz) */
        const REAL8     r,           /**<< distance in SI unit */
        const REAL8     inc,         /**<< inclination angle */
        const REAL8     spin1z,      /**<< z-component of spin-1, dimensionless */
        const REAL8     spin2z       /**<< z-component of spin-2, dimensionless */
     )
{
  /* If either spin > 0.6, model not available, exit */
  if ( spin1z > 0.6 || spin2z > 0.6 )
  {
    printf( "XLAL Error -: Component spin larger than 0.6!\nSEOBNRv1 is only available for spins in the range -1 < a/M < 0.6.\n");
  }

  int i;

  REAL8Vector *values = NULL;

  /* EOB spin vectors used in the Hamiltonian */
  REAL8Vector *sigmaStar = NULL;
  REAL8Vector *sigmaKerr = NULL;
  REAL8       a;
  REAL8       chiS, chiA;

  /* Wrapper spin vectors used to calculate sigmas */
  REAL8Vector s1Vec;
  REAL8Vector s2Vec;
  REAL8       spin1[3] = {0, 0, spin1z};
  REAL8       spin2[3] = {0, 0, spin2z};
  REAL8       s1Data[3], s2Data[3];

  /* Parameters of the system */
  REAL8 m1, m2, mTotal, eta, mTScaled;
  REAL8 amp0;
  REAL8 sSub = 0.0;
  LIGOTimeGPS tc = { 0, 0 };

  /* Dynamics of the system */
  REAL8Vector rVec, phiVec, prVec, pPhiVec;
  REAL8       omega, v, ham;

  /* Cartesian vectors needed to calculate Hamiltonian */
  REAL8Vector cartPosVec, cartMomVec;
  REAL8       cartPosData[3], cartMomData[3];

  /* Signal mode */
  COMPLEX16   hLM;
  REAL8Vector *sigReVec = NULL, *sigImVec = NULL;

  /* Non-quasicircular correction */
  EOBNonQCCoeffs nqcCoeffs;
  COMPLEX16      hNQC;
  REAL8Vector    *ampNQC = NULL, *phaseNQC = NULL;

  /* Ringdown freq used to check the sample rate */
  COMPLEX16Vector modefreqVec;
  COMPLEX16      modeFreq;

  /* Spin-weighted spherical harmonics */
  COMPLEX16  MultSphHarmP;
  COMPLEX16  MultSphHarmM;

  /* We will have to switch to a high sample rate for ringdown attachment */
  REAL8 deltaTHigh;
  UINT4 resampFac;
  UINT4 resampPwr;
  REAL8 resampEstimate;

  /* How far will we have to step back to attach the ringdown? */
  REAL8 tStepBack;
  int  nStepBack;

  /* Dynamics and details of the high sample rate part used to attach the ringdown */
  UINT4 hiSRndx;
  REAL8Vector timeHi, rHi, phiHi, prHi, pPhiHi;
  REAL8Vector *sigReHi = NULL, *sigImHi = NULL;
  REAL8Vector *omegaHi = NULL;

  /* Indices of peak frequency and final point */
  /* Needed to attach ringdown at the appropriate point */
  UINT4 peakIdx = 0, finalIdx = 0;

  /* (2,2) and (2,-2) spherical harmonics needed in (h+,hx) */
  REAL8 y_1, y_2, z1, z2;

  /* Variables for the integrator */
  ark4GSLIntegrator       *integrator = NULL;
  REAL8Array              *dynamics   = NULL;
  REAL8Array              *dynamicsHi = NULL;
  int                    retLen;


  /* Accuracies of adaptive Runge-Kutta integrator */
  const REAL8 EPS_ABS = 1.0e-10;
  const REAL8 EPS_REL = 1.0e-9;

  /**
   * STEP 0) Prepare parameters, including pre-computed coefficients 
   *         for EOB Hamiltonian, flux and waveform  
   */

  /* Parameter structures containing important parameters for the model */
  SpinEOBParams           seobParams;
  SpinEOBHCoeffs          seobCoeffs;
  EOBParams               eobParams;
  FacWaveformCoeffs       hCoeffs;
  NewtonMultipolePrefixes prefixes;

  /* Initialize parameters */
  m1 = m1SI / LAL_MSUN_SI;
  m2 = m2SI / LAL_MSUN_SI;
  mTotal = m1 + m2;
  mTScaled = mTotal * LAL_MTSUN_SI;
  eta    = m1 * m2 / (mTotal*mTotal);

  amp0 = mTotal * LAL_MRSUN_SI / r;

  /* TODO: Insert potentially necessary checks on the arguments */

  /* Calculate the time we will need to step back for ringdown */
  tStepBack = 50. * mTScaled;
  nStepBack = ceil( tStepBack / deltaT );

  /* Calculate the resample factor for attaching the ringdown */
  /* We want it to be a power of 2 */
  /* If deltaT > Mtot/50, reduce deltaT by the smallest power of two for which deltaT < Mtot/50 */
  resampEstimate = 50. * deltaT / mTScaled;
  resampFac = 1;
  
  if ( resampEstimate > 1. )
  {
    resampPwr = (UINT4)ceil( MYlog2( resampEstimate ) );
    while ( resampPwr-- )
    {
      resampFac *= 2u;
    }
  }
  

  /* Allocate the values vector to contain the initial conditions */
  /* Since we have aligned spins, we can use the 4-d vector as in the non-spin case */
  if ( !(values = XLALCreateREAL8Vector( 4 )) )
  {
    	return 0;
  }
  memset ( values->data, 0, values->length * sizeof( REAL8 ));

  /* Set up structures and calculate necessary PN parameters */
  /* Unlike the general case, we only need to calculate these once */
  memset( &seobParams, 0, sizeof(seobParams) );
  memset( &seobCoeffs, 0, sizeof(seobCoeffs) );
  memset( &eobParams, 0, sizeof(eobParams) );
  memset( &hCoeffs, 0, sizeof( hCoeffs ) );
  memset( &prefixes, 0, sizeof( prefixes ) );

  /* Before calculating everything else, check sample freq is high enough */
  modefreqVec.length = 1;
  modefreqVec.data   = &modeFreq;

  if ( XLALSimIMREOBGenerateQNMFreqV2( &modefreqVec, m1, m2, spin1, spin2, 2, 2, 1) == -1 )
  {
    XLALDestroyREAL8Vector( values );
    return -1;
  }


  if ( !(sigmaStar = XLALCreateREAL8Vector( 3 )) )
  {
    XLALDestroyREAL8Vector( values );
    return -1;
  }

  if ( !(sigmaKerr = XLALCreateREAL8Vector( 3 )) )
  {
    XLALDestroyREAL8Vector( sigmaStar );
    XLALDestroyREAL8Vector( values );
    return -1;
  }

  seobParams.alignedSpins = 1;
  seobParams.tortoise     = 1;
  seobParams.sigmaStar    = sigmaStar;
  seobParams.sigmaKerr    = sigmaKerr;
  seobParams.seobCoeffs   = &seobCoeffs;
  seobParams.eobParams    = &eobParams;
  seobParams.Spin1    = XLALCreateREAL8Vector( 3 );
  seobParams.Spin2    = XLALCreateREAL8Vector( 3 );
  eobParams.hCoeffs       = &hCoeffs;
  eobParams.prefixes      = &prefixes;

  eobParams.m1  = m1;
  eobParams.m2  = m2;
  eobParams.eta = eta;

  s1Vec.length = s2Vec.length = 3;
  s1Vec.data   = s1Data;
  s2Vec.data   = s2Data;

  /* copy the spins into the appropriate vectors, and scale them by the mass */
  memcpy( s1Data, spin1, sizeof( s1Data ) );
  memcpy( s2Data, spin2, sizeof( s2Data ) );

  /* Calculate chiS and chiA */


  chiS = 0.5 * (spin1[2] + spin2[2]);
  chiA = 0.5 * (spin1[2] - spin2[2]);

  for( i = 0; i < 3; i++ )
  {
    s1Data[i] *= m1*m1;
    s2Data[i] *= m2*m2;
  }

  cartPosVec.length = cartMomVec.length = 3;
  cartPosVec.data = cartPosData;
  cartMomVec.data = cartMomData;
  memset( cartPosData, 0, sizeof( cartPosData ) );
  memset( cartMomData, 0, sizeof( cartMomData ) );

  /* Populate the initial structures */
  if ( XLALSimIMRSpinEOBCalculateSigmaStar( sigmaStar, m1, m2, &s1Vec, &s2Vec ) == -1 )
  {
    XLALDestroyREAL8Vector( sigmaKerr );
    XLALDestroyREAL8Vector( sigmaStar );
    XLALDestroyREAL8Vector( values );
    return -1;
  }

  if ( XLALSimIMRSpinEOBCalculateSigmaKerr( sigmaKerr, m1, m2, &s1Vec, &s2Vec ) == -1 )
  {
    XLALDestroyREAL8Vector( sigmaKerr );
    XLALDestroyREAL8Vector( sigmaStar );
    XLALDestroyREAL8Vector( values );
    return -1;
  }

  /* Calculate the value of a */
  /* XXX I am assuming that, since spins are aligned, it is okay to just use the z component XXX */
  /* TODO: Check this is actually the way it works in LAL */
  a = 0.0;
  seobParams.a = a = sigmaKerr->data[2];
  /* a set to zero in SEOBNRv1, didn't know yet a good mapping from two physical spins to the test-particle limit Kerr spin */
  if ( XLALSimIMREOBCalcSpinFacWaveformCoefficients( &hCoeffs, m1, m2, eta, /*a*/0.0, chiS, chiA ) == -1 )
  {
    XLALDestroyREAL8Vector( sigmaKerr );
    XLALDestroyREAL8Vector( sigmaStar );
    XLALDestroyREAL8Vector( values );
    return -1;
  }

  if ( XLALSimIMREOBComputeNewtonMultipolePrefixes( &prefixes, eobParams.m1, eobParams.m2 )
         == -1 )
  {
    XLALDestroyREAL8Vector( sigmaKerr );
    XLALDestroyREAL8Vector( sigmaStar );
    XLALDestroyREAL8Vector( values );
    return -1;
  }

  /**
   * STEP 1) Solve for initial conditions
   */

  /* Set the initial conditions. For now we use the generic case */
  /* Can be simplified if spin-aligned initial conditions solver available. The cost of generic code is negligible though. */
  REAL8Vector *tmpValues = XLALCreateREAL8Vector( 14 );
  if ( !tmpValues )
  {
    XLALDestroyREAL8Vector( sigmaKerr );
    XLALDestroyREAL8Vector( sigmaStar );
    XLALDestroyREAL8Vector( values );
   	return -1;
  }

  memset( tmpValues->data, 0, tmpValues->length * sizeof( REAL8 ) );

  /* We set inc zero here to make it easier to go from Cartesian to spherical coords */
  /* No problem setting inc to zero in solving spin-aligned initial conditions. */
  /* inc is not zero in generating the final h+ and hx */
  if ( XLALSimIMRSpinEOBInitialConditions( tmpValues, m1, m2, fMin, e0, 0, s1Data, s2Data, &seobParams ) == XLAL_FAILURE )
  {
    XLALDestroyREAL8Vector( tmpValues );
    XLALDestroyREAL8Vector( sigmaKerr );
    XLALDestroyREAL8Vector( sigmaStar );
    XLALDestroyREAL8Vector( values );
    return -1;
  }

  /* Taken from Andrea's code */
  /* Now convert to Spherical */
  /* The initial conditions code returns Cartesian components of four vectors x, p, S1 and S2,
   * in the special case that the binary starts on the x-axis and the two spins are aligned
   * with the orbital angular momentum along the z-axis.
   * Therefore, in spherical coordinates the initial conditions are
   * r = x; phi = 0.; pr = px; pphi = r * py.
   */
  values->data[0] = tmpValues->data[0];
  values->data[1] = 0.;
  values->data[2] = tmpValues->data[3];
  values->data[3] = tmpValues->data[0] * tmpValues->data[4];

  /* Now compute the spinning H coefficients and store them in seobCoeffs */
  if ( XLALSimIMRCalculateSpinEOBHCoeffs( &seobCoeffs, eta, a ) == -1 )
  {    
    XLALDestroyREAL8Vector( tmpValues );
    XLALDestroyREAL8Vector( sigmaKerr );
    XLALDestroyREAL8Vector( sigmaStar );
    XLALDestroyREAL8Vector( values );
    return -1;
  }

  /**
   * STEP 2) Evolve EOB trajectory until reaching the peak of orbital frequency
   */

  /* Now we have the initial conditions, we can initialize the adaptive integrator */
  /* XLALAdaptiveRungeKutta4Init is in Panyicomm_elip.cpp */
  /* XLALSpinAlignedHcapDerivative is in Panyicomm_elip.cpp */
  /* XLALEOBSpinAlignedStopCondition is in Panyicomm_elip.cpp */
  if (!(integrator = XLALAdaptiveRungeKutta4Init(4, XLALSpinAlignedHcapDerivative, XLALEOBSpinAlignedStopCondition, EPS_ABS, EPS_REL)))
  {
    XLALDestroyREAL8Vector( values );
  }

  integrator->stopontestonly = 1;
  integrator->retries = 1;

  retLen = XLALAdaptiveRungeKutta4( integrator, &seobParams, values->data, 0., 20./mTScaled, deltaT/mTScaled, &dynamics );
  if ( retLen == XLAL_FAILURE )
  {
    printf("Error!");
  }

  /* Set up pointers to the dynamics */
  rVec.length = phiVec.length = prVec.length = pPhiVec.length = retLen;
  rVec.data    = dynamics->data+retLen;
  phiVec.data  = dynamics->data+2*retLen;
  prVec.data   = dynamics->data+3*retLen;
  pPhiVec.data = dynamics->data+4*retLen;

  /**
   * STEP 3) Step back in time by tStepBack and volve EOB trajectory again 
   *         using high sampling rate, stop at 0.3M out of the "EOB horizon".
   */

  /* Set up the high sample rate integration */
  hiSRndx = retLen - nStepBack;
  deltaTHigh = deltaT / (REAL8)resampFac;

  values->data[0] = rVec.data[hiSRndx];
  values->data[1] = phiVec.data[hiSRndx];
  values->data[2] = prVec.data[hiSRndx];
  values->data[3] = pPhiVec.data[hiSRndx];
  /* For HiSR evolution, we stop at a radius 0.3M from the deformed Kerr singularity, 
   * or when any derivative of Hamiltonian becomes nan */
  integrator->stop = XLALSpinAlignedHiSRStopCondition;

  retLen = XLALAdaptiveRungeKutta4( integrator, &seobParams, values->data, 0., 20./mTScaled, deltaTHigh/mTScaled, &dynamicsHi );

  if ( retLen == XLAL_FAILURE )
  {
    printf("Error!");
  }


  /* Set up pointers to the dynamics */
  rHi.length = phiHi.length = prHi.length = pPhiHi.length = timeHi.length = retLen;
  timeHi.data = dynamicsHi->data;
  rHi.data    = dynamicsHi->data+retLen;
  phiHi.data  = dynamicsHi->data+2*retLen;
  prHi.data   = dynamicsHi->data+3*retLen;
  pPhiHi.data = dynamicsHi->data+4*retLen;

  /* Allocate the high sample rate vectors */
  sigReHi  = XLALCreateREAL8Vector( retLen + (UINT4)ceil( 20 / ( cimag(modeFreq) * deltaTHigh )) );
  sigImHi  = XLALCreateREAL8Vector( retLen + (UINT4)ceil( 20 / ( cimag(modeFreq) * deltaTHigh )) );
  omegaHi  = XLALCreateREAL8Vector( retLen + (UINT4)ceil( 20 / ( cimag(modeFreq) * deltaTHigh )) );
  ampNQC   = XLALCreateREAL8Vector( retLen );
  phaseNQC = XLALCreateREAL8Vector( retLen );

  if ( !sigReHi || !sigImHi || !omegaHi || !ampNQC || !phaseNQC )
  {
    printf("Error!");
  }

  memset( sigReHi->data, 0, sigReHi->length * sizeof( sigReHi->data[0] ));
  memset( sigImHi->data, 0, sigImHi->length * sizeof( sigImHi->data[0] ));

  /* Populate the high SR waveform */
  REAL8 omegaOld = 0.0;
  INT4  phaseCounter = 0;

  for ( i = 0; i < retLen; i++ )
  {
    values->data[0] = rHi.data[i];
    values->data[1] = phiHi.data[i];
    values->data[2] = prHi.data[i];
    values->data[3] = pPhiHi.data[i];

    omegaHi->data[i] = omega = XLALSimIMRSpinAlignedEOBCalcOmega( values->data, &seobParams );
    v = MYcbrt( omega );

    /* Calculate the value of the Hamiltonian */
    cartPosVec.data[0] = values->data[0];
    cartMomVec.data[0] = values->data[2];
    cartMomVec.data[1] = values->data[3] / values->data[0];

    ham = XLALSimIMRSpinEOBHamiltonian( eta, &cartPosVec, &cartMomVec, sigmaKerr, sigmaStar, seobParams.tortoise, &seobCoeffs );

    if ( XLALSimIMRSpinEOBGetSpinFactorizedWaveform( &hLM, values, v, ham, 2, 2, &seobParams )
           == XLAL_FAILURE )
    {
      /* TODO: Clean-up */
      printf("Error!");
    }

    ampNQC->data[i]  = cabs( hLM );
    sigReHi->data[i] = (REAL4)(amp0 * creal(hLM));
    sigImHi->data[i] = (REAL4)(amp0 * cimag(hLM));
    phaseNQC->data[i]= carg( hLM ) + phaseCounter * LAL_TWOPI;

    if ( i && phaseNQC->data[i] > phaseNQC->data[i-1] )
    {
      phaseCounter--;
      phaseNQC->data[i] -= LAL_TWOPI;
    }

    if ( omega <= omegaOld && !peakIdx )
    {
      peakIdx = i;
    }
    omegaOld = omega;
  }
  finalIdx = retLen - 1;

  /**
   * STEP 4) Locate the peak of orbital frequency for NQC and QNM calculations
   */

  /* Stuff to find the actual peak time */
  gsl_spline    *spline = NULL;
  gsl_interp_accel *acc = NULL;
  REAL8 omegaDeriv1; //, omegaDeriv2;
  REAL8 time1, time2;
  REAL8 timePeak, timewavePeak = 0., omegaDerivMid;
  REAL8 sigAmpSqHi = 0., oldsigAmpSqHi = 0.;
  INT4  peakCount = 0;

  spline = gsl_spline_alloc( gsl_interp_cspline, retLen );
  acc    = gsl_interp_accel_alloc();

  time1 = dynamicsHi->data[peakIdx];

  gsl_spline_init( spline, dynamicsHi->data, omegaHi->data, retLen );
  omegaDeriv1 = gsl_spline_eval_deriv( spline, time1, acc );
  if ( omegaDeriv1 > 0. )
  {
    time2 = dynamicsHi->data[peakIdx+1];
  }
  else
  {
    time2 = time1;
    time1 = dynamicsHi->data[peakIdx-1];
    peakIdx--;
    omegaDeriv1 = gsl_spline_eval_deriv( spline, time1, acc );
  }

  do
  {
    timePeak = ( time1 + time2 ) / 2.;
    omegaDerivMid = gsl_spline_eval_deriv( spline, timePeak, acc );

    if ( omegaDerivMid * omegaDeriv1 < 0.0 )
    {
      time2 = timePeak;
    }
    else
    {
      omegaDeriv1 = omegaDerivMid;
      time1 = timePeak;
    }
  }
  while ( time2 - time1 > 1.0e-5 );

  /* Having located the peak of orbital frequency, we set time and phase of coalescence */
  XLALGPSAdd( &tc, -mTScaled * (dynamics->data[hiSRndx] + timePeak));
  gsl_spline_init( spline, dynamicsHi->data, phiHi.data, retLen );
  sSub = gsl_spline_eval( spline, timePeak, acc ) - phiC;
  gsl_spline_free( spline );
  gsl_interp_accel_free( acc );
  /* Apply phiC to hi-sampling waveforms */
  REAL8 thisReHi, thisImHi;
  REAL8 csSub2 = cos(2.0 * sSub);
  REAL8 ssSub2 = sin(2.0 * sSub);
  for ( i = 0; i < retLen; i++)
  {
    thisReHi = sigReHi->data[i];
    thisImHi = sigImHi->data[i];
    sigReHi->data[i] =   thisReHi * csSub2 - thisImHi * ssSub2; //set the phase of waveform at peak orbital frequancy as 0, C*exp(i 2*sSub)
    sigImHi->data[i] =   thisReHi * ssSub2 + thisImHi * csSub2; 
  }

  /**
   * STEP 5) Calculate NQC correction using hi-sampling data
   */

  /* Calculate nonspin and amplitude NQC coefficients from fits and interpolation table */
  if ( XLALSimIMRGetEOBCalibratedSpinNQC( &nqcCoeffs, 2, 2, eta, a ) == XLAL_FAILURE )
  {
    printf("Error(%d)", XLAL_EFUNC );
  }

  /* Calculate phase NQC coefficients */
  if ( XLALSimIMRSpinEOBCalculateNQCCoefficients( ampNQC, phaseNQC, &rHi, &prHi, omegaHi,
          2, 2, timePeak, deltaTHigh/mTScaled, eta, a, &nqcCoeffs ) == XLAL_FAILURE )
  {
	  printf("Error(%d)", XLAL_EFUNC );
  }

  /* Calculate the time of amplitude peak. Despite the name, this is in fact the shift in peak time from peak orb freq time */
  timewavePeak = XLALSimIMREOBGetNRSpinPeakDeltaT(2, 2, eta,  a);
 
  /* Apply to the high sampled part */
  for ( i = 0; i < retLen; i++ )
  {
    values->data[0] = rHi.data[i];
    values->data[1] = phiHi.data[i] - sSub;
    values->data[2] = prHi.data[i];
    values->data[3] = pPhiHi.data[i];

    if ( XLALSimIMREOBNonQCCorrection( &hNQC, values, omegaHi->data[i], &nqcCoeffs ) == XLAL_FAILURE )
    {
		printf("Error(%d)", XLAL_EFUNC );
    }

    hLM = sigReHi->data[i];
    hLM += I * sigImHi->data[i];

    hLM *= hNQC;
      {
        double r,phi,pr,pphi,rdot,phidot;
        values->data[1] += sSub;
        r = values->data[0];
        phi = values->data[1];
	pr = values->data[2];
	pphi = values->data[3];
	double dydt[4],ttmp=0;
        XLALSpinAlignedHcapDerivative(ttmp, values->data, dydt, &seobParams);
        values->data[1] -= sSub;
        rdot = dydt[0];
        phidot = dydt[1];

        double x1,x2,x3;
        x1 = r*cos(phi);
        x2 = r*sin(phi);
        x3 = 0;
        double v1,v2,v3;
        v1 = rdot*cos(phi)-r*phidot*sin(phi);
        v2 = rdot*sin(phi)+r*phidot*cos(phi);
        v3 = 0;

        double hp,hc;
        PNwaveformPRD544813rdotc_22mode(&hp,&hc,x1,x2,x3,v1,v2,v3,eta);

	hLM = hLM*exp(-I*2.0*sSub);
#if 1
        hLM += amp0*(hp+I*hc);
#else
        hLM += amp0*(hp+I*hc)/cabs(hNQC);  // relax by hNQC for late inspiral
#endif
	hLM = hLM*exp(I*2.0*sSub);
      }
    sigReHi->data[i] = (REAL4) creal(hLM);
    sigImHi->data[i] = (REAL4) cimag(hLM);
    sigAmpSqHi = creal(hLM)*creal(hLM)+cimag(hLM)*cimag(hLM);
    if (sigAmpSqHi < oldsigAmpSqHi && peakCount == 0 && (i-1)*deltaTHigh/mTScaled < timePeak - timewavePeak) 
    {
      timewavePeak = (i-1)*deltaTHigh/mTScaled;
      peakCount += 1;
    }
    oldsigAmpSqHi = sigAmpSqHi;
  }
  if (timewavePeak < 1.0e-16 || peakCount == 0)
  {
    /* NOTE: instead of looking for the actual peak, use the calibrated value,    */
    /*       ignoring the error in using interpolated NQC instead of iterated NQC */
    timewavePeak = timePeak - timewavePeak;
  }
  
  /**
   * STEP 6) Calculate QNM excitation coefficients using hi-sampling data
   */
  
  /* Attach the ringdown at the time of amplitude peak */
  REAL8 combSize = 7.5; /* Eq. 34 */
  REAL8 timeshiftPeak;
  timeshiftPeak = timePeak - timewavePeak;


  REAL8Vector *rdMatchPoint = XLALCreateREAL8Vector( 3 );
  if ( !rdMatchPoint )
  {
    printf("Error(%d)", XLAL_ENOMEM );
  }

  if ( combSize > timePeak - timeshiftPeak )
  {
    printf( "The comb size looks to be too big!!!\n" );
  }

  rdMatchPoint->data[0] = combSize < timePeak - timeshiftPeak ? timePeak - timeshiftPeak - combSize : 0;
  rdMatchPoint->data[1] = timePeak - timeshiftPeak;
  rdMatchPoint->data[2] = dynamicsHi->data[finalIdx];

  //XLALSimIMREOBHybridAttachRingdown locates in the current file
  if ( XLALSimIMREOBHybridAttachRingdown( sigReHi, sigImHi, 2, 2,
              deltaTHigh, m1, m2, spin1[0], spin1[1], spin1[2], spin2[0], spin2[1], spin2[2],
              &timeHi, rdMatchPoint)
          == XLAL_FAILURE ) 
  {
  	
    printf("Error(%d)", XLAL_EFUNC );
  }

  /**
   * STEP 7) Generate full inspiral waveform using desired sampling frequency
   */

  /* Now create vectors at the correct sample rate, and compile the complete waveform */
  sigReVec = XLALCreateREAL8Vector( rVec.length + (UINT4)ceil( sigReHi->length / (double)resampFac ) );
  sigImVec = XLALCreateREAL8Vector( sigReVec->length );

  memset( sigReVec->data, 0, sigReVec->length * sizeof( REAL8 ) );
  memset( sigImVec->data, 0, sigImVec->length * sizeof( REAL8 ) );
 
  /* Generate full inspiral waveform using desired sampling frequency */
  /* TODO - Check vectors were allocated */
  for ( i = 0; i < (INT4)rVec.length; i++ )
  {
    values->data[0] = rVec.data[i];
    values->data[1] = phiVec.data[i] - sSub;
    values->data[2] = prVec.data[i];
    values->data[3] = pPhiVec.data[i];

    omega = XLALSimIMRSpinAlignedEOBCalcOmega( values->data, &seobParams );
    v = MYcbrt( omega );

    /* Calculate the value of the Hamiltonian */
    cartPosVec.data[0] = values->data[0];
    cartMomVec.data[0] = values->data[2];
    cartMomVec.data[1] = values->data[3] / values->data[0];

    ham = XLALSimIMRSpinEOBHamiltonian( eta, &cartPosVec, &cartMomVec, sigmaKerr, sigmaStar, seobParams.tortoise, &seobCoeffs );

    if ( XLALSimIMRSpinEOBGetSpinFactorizedWaveform( &hLM, values, v, ham, 2, 2, &seobParams )
           == XLAL_FAILURE )
    {
      /* TODO: Clean-up */
	  printf("Error(%d)", XLAL_EFUNC );
    }

    if ( XLALSimIMREOBNonQCCorrection( &hNQC, values, omega, &nqcCoeffs ) == XLAL_FAILURE )
    {
		printf("Error(%d)", XLAL_EFUNC );
    }

    hLM *= hNQC;

    sigReVec->data[i] = amp0 * creal(hLM);
    sigImVec->data[i] = amp0 * cimag(hLM);
  }

  /**
   * STEP 8) Generate full IMR modes -- attaching ringdown to inspiral
   */

  /* Attach the ringdown part to the inspiral */
  for ( i = 0; i < (INT4)(sigReHi->length / resampFac); i++ )
  {
    sigReVec->data[i+hiSRndx] = sigReHi->data[i*resampFac];  // right hand side include high resolution inspiral and ringdown
    sigImVec->data[i+hiSRndx] = sigImHi->data[i*resampFac];
  }

  /**
   * STEP 9) Generate full IMR hp and hx waveforms
   */
  
  /* For now, let us just try to create a waveform */
  REAL8TimeSeries *hPlusTS  = XLALCreateREAL8TimeSeries( "H_PLUS", &tc, 0.0, deltaT, &lalStrainUnit, sigReVec->length );
  REAL8TimeSeries *hCrossTS = XLALCreateREAL8TimeSeries( "H_CROSS", &tc, 0.0, deltaT, &lalStrainUnit, sigImVec->length );

  /* TODO change to using XLALSimAddMode function to combine modes */
  /* For now, calculate -2Y22 * h22 + -2Y2-2 * h2-2 directly (all terms complex) */
  /* Compute spin-weighted spherical harmonics and generate waveform */
  REAL8 coa_phase = 0.0;

  MultSphHarmP = XLALSpinWeightedSphericalHarmonic( inc, coa_phase, -2, 2, 2 ); // -2Y22(inc, coa_phase)
  MultSphHarmM = XLALSpinWeightedSphericalHarmonic( inc, coa_phase, -2, 2, -2 ); // -2Y2-2(inc, coa_phase)

  y_1 =   creal(MultSphHarmP) + creal(MultSphHarmM);
  y_2 =   cimag(MultSphHarmM) - cimag(MultSphHarmP) ;
  z1 = -  cimag(MultSphHarmM) - cimag(MultSphHarmP) ;
  z2 =    creal(MultSphHarmM) - creal(MultSphHarmP);

  for ( i = 0; i < (INT4)sigReVec->length; i++ )
  {
    REAL8 x1 = sigReVec->data[i];
    REAL8 x2 = sigImVec->data[i];
#if 0
    hPlusTS->data->data[i]  = x1;
    hCrossTS->data->data[i] = x2;
#else
    hPlusTS->data->data[i]  = y_1*x1 + y_2*x2;
    hCrossTS->data->data[i] = z1*x1 + z2*x2;
#endif
  }

  /* Point the output pointers to the relevant time series and return */
  (*hplus)  = hPlusTS;
  (*hcross) = hCrossTS;

  /* Free memory */
  XLALDestroyREAL8Vector( tmpValues );
  XLALDestroyREAL8Vector( sigmaKerr );
  XLALDestroyREAL8Vector( sigmaStar );
  XLALDestroyREAL8Vector( values );
  XLALDestroyREAL8Vector( rdMatchPoint );
  XLALDestroyREAL8Vector( ampNQC );
  XLALDestroyREAL8Vector( phaseNQC );
  XLALDestroyREAL8Vector( sigReVec );
  XLALDestroyREAL8Vector( sigImVec );
  XLALAdaptiveRungeKutta4Free( integrator );
  XLALDestroyREAL8Array( dynamics );
  XLALDestroyREAL8Array( dynamicsHi );
  XLALDestroyREAL8Vector( sigReHi );
  XLALDestroyREAL8Vector( sigImHi );
  XLALDestroyREAL8Vector( omegaHi );

  return XLAL_SUCCESS;
}


/**
 * The main workhorse function for performing the ringdown attachment for EOB
 * models EOBNRv2 and SEOBNRv1. This is the function which gets called by the 
 * code generating the full IMR waveform once generation of the inspiral part
 * has been completed.
 * The ringdown is attached using the hybrid comb matching detailed in 
 * The method is describe in Sec. II C of Pan et al. PRD 84, 124052 (2011), 
 * specifically Eqs. 30 - 32.. Further details of the
 * implementation of the found in the DCC document T1100433.
 * In SEOBNRv1, the last physical overtone is replace by a pseudoQNM. See 
 * Taracchini et al. PRD 86, 024011 (2012) for details.
 * STEP 1) Get mass and spin of the final black hole and the complex ringdown frequencies
 * STEP 2) Based on least-damped-mode decay time, allocate memory for rigndown waveform
 * STEP 3) Get values and derivatives of inspiral waveforms at matching comb points
 * STEP 4) Solve QNM coefficients and generate ringdown waveforms
 * STEP 5) Stitch inspiral and ringdown waveoforms
 */
 INT4 XLALSimIMREOBHybridAttachRingdown(
  REAL8Vector *signal1,    /**<< OUTPUT, Real of inspiral waveform to which we attach ringdown */
  REAL8Vector *signal2,    /**<< OUTPUT, Imag of inspiral waveform to which we attach ringdown */
  const INT4   l,          /**<< Current mode l */
  const INT4   m,          /**<< Current mode m */
  const REAL8  dt,         /**<< Sample time step (in seconds) */
  const REAL8  mass1,      /**<< First component mass (in Solar masses) */
  const REAL8  mass2,      /**<< Second component mass (in Solar masses) */
  const REAL8  spin1x,     /**<<The spin of the first object; only needed for spin waveforms */
  const REAL8  spin1y,     /**<<The spin of the first object; only needed for spin waveforms */
  const REAL8  spin1z,     /**<<The spin of the first object; only needed for spin waveforms */
  const REAL8  spin2x,     /**<<The spin of the second object; only needed for spin waveforms */
  const REAL8  spin2y,     /**<<The spin of the second object; only needed for spin waveforms */
  const REAL8  spin2z,     /**<<The spin of the second object; only needed for spin waveforms */
  REAL8Vector *timeVec,    /**<< Vector containing the time values */
  REAL8Vector *matchrange /**<< Time values chosen as points for performing comb matching */
  )
{

      COMPLEX16Vector *modefreqs;
      UINT4 Nrdwave;
      UINT4 j;

      UINT4 nmodes;
      REAL8Vector		*rdwave1;
      REAL8Vector		*rdwave2;
      REAL8Vector		*rinspwave;
      REAL8Vector		*dinspwave;
      REAL8Vector		*ddinspwave;
      REAL8VectorSequence	*inspwaves1;
      REAL8VectorSequence	*inspwaves2;
      REAL8 eta, a, NRPeakOmega22; /* To generate pQNM frequency */
      REAL8 mTot; /* In geometric units */
      REAL8 spin1[3] = { spin1x, spin1y, spin1z };
      REAL8 spin2[3] = { spin2x, spin2y, spin2z };
      REAL8 finalMass, finalSpin;

      mTot  = (mass1 + mass2) * LAL_MTSUN_SI;
      eta       = mass1 * mass2 / ( (mass1 + mass2) * (mass1 + mass2) );

      /**
       * STEP 1) Get mass and spin of the final black hole and the complex ringdown frequencies
       */

      /* Create memory for the QNM frequencies */
      nmodes = 8;
      modefreqs = XLALCreateCOMPLEX16Vector( nmodes );
      if ( !modefreqs )
      {
        printf("Error(%d)", XLAL_ENOMEM );
      }

      if ( XLALSimIMREOBGenerateQNMFreqV2( modefreqs, mass1, mass2, spin1, spin2, l, m, nmodes ) == XLAL_FAILURE )
      {
        XLALDestroyCOMPLEX16Vector( modefreqs );
        
        printf("Error(%d)", XLAL_EFUNC );
      }

      /* Call XLALSimIMREOBFinalMassSpin() to get mass and spin of the final black hole */
      if ( XLALSimIMREOBFinalMassSpin(&finalMass, &finalSpin, mass1, mass2, spin1, spin2) == XLAL_FAILURE )
      {
		  printf("Error(%d)", XLAL_EFUNC );
      }

          /* Replace the last QNM with pQNM */
          /* We assume aligned/antialigned spins here */
      a  = (spin1[2] + spin2[2]) / 2. * (1.0 - 2.0 * eta) + (spin1[2] - spin2[2]) / 2. * (mass1 - mass2) / (mass1 + mass2);
      NRPeakOmega22 = GetNRSpinPeakOmega( l, m, eta, a ) / mTot;
      modefreqs->data[7] = (NRPeakOmega22/finalMass + creal(modefreqs->data[0])) / 2.;
      modefreqs->data[7] += I * 10./3. * cimag(modefreqs->data[0]);

      /* Ringdown signal length: 10 times the decay time of the n=0 mode */
      Nrdwave = (INT4) (EOB_RD_EFOLDS / cimag(modefreqs->data[0]) / dt);

      /* Check the value of attpos, to prevent memory access problems later */
      if ( matchrange->data[0] * mTot / dt < 5 || matchrange->data[1]*mTot/dt > matchrange->data[2] *mTot/dt - 2 )
      {
        printf( "More inspiral points needed for ringdown matching.\n" );
        XLALDestroyCOMPLEX16Vector( modefreqs );
		
		printf("Error(%d)", XLAL_EFAILED );
      }

      /**
       * STEP 2) Based on least-damped-mode decay time, allocate memory for rigndown waveform
       */

      /* Create memory for the ring-down and full waveforms, and derivatives of inspirals */

      rdwave1 = XLALCreateREAL8Vector( Nrdwave );
      rdwave2 = XLALCreateREAL8Vector( Nrdwave );
      rinspwave = XLALCreateREAL8Vector( 6 );
      dinspwave = XLALCreateREAL8Vector( 6 );
      ddinspwave = XLALCreateREAL8Vector( 6 );
      inspwaves1 = XLALCreateREAL8VectorSequence( 3, 6 );
      inspwaves2 = XLALCreateREAL8VectorSequence( 3, 6 );

      /* Check memory was allocated */
      if ( !rdwave1 || !rdwave2 || !rinspwave || !dinspwave 
	   || !ddinspwave || !inspwaves1 || !inspwaves2 )
      {
        XLALDestroyCOMPLEX16Vector( modefreqs );
        if (rdwave1)    XLALDestroyREAL8Vector( rdwave1 );
        if (rdwave2)    XLALDestroyREAL8Vector( rdwave2 );
        if (rinspwave)  XLALDestroyREAL8Vector( rinspwave );
        if (dinspwave)  XLALDestroyREAL8Vector( dinspwave );
        if (ddinspwave) XLALDestroyREAL8Vector( ddinspwave );
        if (inspwaves1) XLALDestroyREAL8VectorSequence( inspwaves1 );
        if (inspwaves2) XLALDestroyREAL8VectorSequence( inspwaves2 );
        printf("Error(%d)", XLAL_ENOMEM );
      }

      memset( rdwave1->data, 0, rdwave1->length * sizeof( REAL8 ) );
      memset( rdwave2->data, 0, rdwave2->length * sizeof( REAL8 ) );

      /**
       * STEP 3) Get values and derivatives of inspiral waveforms at matching comb points
       */

      /* Generate derivatives of the last part of inspiral waves */
      /* Get derivatives of signal1 */
      if ( XLALGenerateHybridWaveDerivatives( rinspwave, dinspwave, ddinspwave, timeVec, signal1, 
			matchrange, dt, mass1, mass2 ) == XLAL_FAILURE )
      {
        XLALDestroyCOMPLEX16Vector( modefreqs );
        XLALDestroyREAL8Vector( rdwave1 );
        XLALDestroyREAL8Vector( rdwave2 );
        XLALDestroyREAL8Vector( rinspwave );
        XLALDestroyREAL8Vector( dinspwave );
        XLALDestroyREAL8Vector( ddinspwave );
        XLALDestroyREAL8VectorSequence( inspwaves1 );
        XLALDestroyREAL8VectorSequence( inspwaves2 );
		printf("Error(%d)", XLAL_EFUNC );
      }
      for (j = 0; j < 6; j++)
      {
	    inspwaves1->data[j] = rinspwave->data[j];
	    inspwaves1->data[j + 6] = dinspwave->data[j];
	    inspwaves1->data[j + 12] = ddinspwave->data[j];
      }

      /* Get derivatives of signal2 */
      if ( XLALGenerateHybridWaveDerivatives( rinspwave, dinspwave, ddinspwave, timeVec, signal2, 
			matchrange, dt, mass1, mass2 ) == XLAL_FAILURE )
      {
        XLALDestroyCOMPLEX16Vector( modefreqs );
        XLALDestroyREAL8Vector( rdwave1 );
        XLALDestroyREAL8Vector( rdwave2 );
        XLALDestroyREAL8Vector( rinspwave );
        XLALDestroyREAL8Vector( dinspwave );
        XLALDestroyREAL8Vector( ddinspwave );
        XLALDestroyREAL8VectorSequence( inspwaves1 );
        XLALDestroyREAL8VectorSequence( inspwaves2 );
		printf("Error(%d)", XLAL_EFUNC );
      }
      for (j = 0; j < 6; j++)
      {
	    inspwaves2->data[j] = rinspwave->data[j];
	    inspwaves2->data[j + 6] = dinspwave->data[j];
	    inspwaves2->data[j + 12] = ddinspwave->data[j];
      }


      /**
       * STEP 4) Solve QNM coefficients and generate ringdown waveforms
       */

      /* Generate ring-down waveforms */
      if ( XLALSimIMREOBHybridRingdownWave( rdwave1, rdwave2, dt, mass1, mass2, inspwaves1, inspwaves2,
			  modefreqs, matchrange ) == XLAL_FAILURE )
      {
        XLALDestroyCOMPLEX16Vector( modefreqs );
        XLALDestroyREAL8Vector( rdwave1 );
        XLALDestroyREAL8Vector( rdwave2 );
        XLALDestroyREAL8Vector( rinspwave );
        XLALDestroyREAL8Vector( dinspwave );
        XLALDestroyREAL8Vector( ddinspwave );
        XLALDestroyREAL8VectorSequence( inspwaves1 );
        XLALDestroyREAL8VectorSequence( inspwaves2 );
		printf("Error(%d)", XLAL_EFUNC );
      }

      /**
       * STEP 5) Stitch inspiral and ringdown waveoforms
       */

      /* Generate full waveforms, by stitching inspiral and ring-down waveforms */
      UINT4 attachIdx = (UINT4)(matchrange->data[1] * mTot / dt);
      for (j = 1; j < Nrdwave; ++j)
      {
	    signal1->data[j + attachIdx] = rdwave1->data[j];
	    signal2->data[j + attachIdx] = rdwave2->data[j];
      }

      memset( signal1->data+Nrdwave+attachIdx, 0, (signal1->length - Nrdwave - attachIdx)*sizeof(REAL8) );
      memset( signal2->data+Nrdwave+attachIdx, 0, (signal2->length - Nrdwave - attachIdx)*sizeof(REAL8) );

      /* Free memory */
      XLALDestroyCOMPLEX16Vector( modefreqs );
      XLALDestroyREAL8Vector( rdwave1 );
      XLALDestroyREAL8Vector( rdwave2 );
      XLALDestroyREAL8Vector( rinspwave );
      XLALDestroyREAL8Vector( dinspwave );
      XLALDestroyREAL8Vector( ddinspwave );
      XLALDestroyREAL8VectorSequence( inspwaves1 );
      XLALDestroyREAL8VectorSequence( inspwaves2 );

      return XLAL_SUCCESS;
}

/**
 * Generates the ringdown wave associated with the given real
 * and imaginary parts of the inspiral waveform. The parameters of 
 * the ringdown, such as amplitude and phase offsets, are determined
 * by solving the linear equations defined in the DCC document T1100433.
 * In the linear equations Ax=y, 
 * A is a 16-by-16 matrix depending on QNM (complex) frequencies,
 * x is a 16-d vector of the 8 unknown complex QNM amplitudes,
 * y is a 16-d vector depending on inspiral-plunge waveforms and their derivatives near merger.
 */ 
 INT4 XLALSimIMREOBHybridRingdownWave(
  REAL8Vector          *rdwave1,   /**<< OUTPUT, Real part of ringdown waveform */
  REAL8Vector          *rdwave2,   /**<< OUTPUT, Imag part of ringdown waveform */
  const REAL8           dt,        /**<< Sampling interval */
  const REAL8           mass1,     /**<< First component mass (in Solar masses) */
  const REAL8           mass2,     /**<< Second component mass (in Solar masses) */
  REAL8VectorSequence  *inspwave1, /**<< Values and derivs of real part inspiral waveform */
  REAL8VectorSequence  *inspwave2, /**<< Values and derivs of imag part inspiral waveform */
  COMPLEX16Vector      *modefreqs, /**<< Complex freqs of ringdown (scaled by total mass) */
  REAL8Vector          *matchrange /**<< Times which determine the comb of ringdown attachment */
  )
{

  /* XLAL error handling */
  INT4 errcode = XLAL_SUCCESS;

  /* For checking GSL return codes */
  INT4 gslStatus;

  UINT4 i, j, k, nmodes = 8;

  /* Sampling rate from input */
  REAL8 t1, t2, t3, t4, t5, rt;
  gsl_matrix *coef;
  gsl_vector *hderivs;
  gsl_vector *x;
  gsl_permutation *p;
  REAL8Vector *modeamps;
  int s;
  REAL8 tj;
  REAL8 m;

  /* mass in geometric units */
  m  = (mass1 + mass2) * LAL_MTSUN_SI;
  t5 = (matchrange->data[0] - matchrange->data[1]) * m;
  rt = -t5 / 5.;

  t4 = t5 + rt;
  t3 = t4 + rt;
  t2 = t3 + rt;
  t1 = t2 + rt;
  
  if ( inspwave1->length != 3 || inspwave2->length != 3 ||
		modefreqs->length != nmodes )
  {
    printf("Error(%d)", XLAL_EBADLEN );
  }

  /* Solving the linear system for QNMs amplitude coefficients using gsl routine */
  /* Initiate matrices and supporting variables */
  coef = (gsl_matrix *) gsl_matrix_alloc(2 * nmodes, 2 * nmodes);
  hderivs = (gsl_vector *) gsl_vector_alloc(2 * nmodes);
   x = (gsl_vector *) gsl_vector_alloc(2 * nmodes);
  p = (gsl_permutation *) gsl_permutation_alloc(2 * nmodes);

  /* Check all matrices and variables were allocated */
  if ( !coef || !hderivs || !x || !p )
  {
    if (coef)    gsl_matrix_free(coef);
    if (hderivs) gsl_vector_free(hderivs);
    if (x)       gsl_vector_free(x);
    if (p)       gsl_permutation_free(p);
    printf("Error(%d)", XLAL_ENOMEM );
  }

  /* Define the linear system Ax=y */
  /* Matrix A (2*n by 2*n) has block symmetry. Define half of A here as "coef" */
  /* The half of A defined here corresponds to matrices M1 and -M2 in the DCC document T1100433 */ 
  /* Define y here as "hderivs" */
  for (i = 0; i < nmodes; ++i)
  {
	gsl_matrix_set(coef, 0, i, 1);
	gsl_matrix_set(coef, 1, i, - cimag(modefreqs->data[i]));
	gsl_matrix_set(coef, 2, i, exp(-cimag(modefreqs->data[i])*t1) * cos(creal(modefreqs->data[i])*t1));
	gsl_matrix_set(coef, 3, i, exp(-cimag(modefreqs->data[i])*t2) * cos(creal(modefreqs->data[i])*t2));
	gsl_matrix_set(coef, 4, i, exp(-cimag(modefreqs->data[i])*t3) * cos(creal(modefreqs->data[i])*t3));
	gsl_matrix_set(coef, 5, i, exp(-cimag(modefreqs->data[i])*t4) * cos(creal(modefreqs->data[i])*t4));
	gsl_matrix_set(coef, 6, i, exp(-cimag(modefreqs->data[i])*t5) * cos(creal(modefreqs->data[i])*t5));
	gsl_matrix_set(coef, 7, i, exp(-cimag(modefreqs->data[i])*t5) * 
				      (-cimag(modefreqs->data[i]) * cos(creal(modefreqs->data[i])*t5)
				       -creal(modefreqs->data[i]) * sin(creal(modefreqs->data[i])*t5)));
	gsl_matrix_set(coef, 8, i, 0);
	gsl_matrix_set(coef, 9, i, - creal(modefreqs->data[i]));
	gsl_matrix_set(coef, 10, i, -exp(-cimag(modefreqs->data[i])*t1) * sin(creal(modefreqs->data[i])*t1));
	gsl_matrix_set(coef, 11, i, -exp(-cimag(modefreqs->data[i])*t2) * sin(creal(modefreqs->data[i])*t2));
	gsl_matrix_set(coef, 12, i, -exp(-cimag(modefreqs->data[i])*t3) * sin(creal(modefreqs->data[i])*t3));
	gsl_matrix_set(coef, 13, i, -exp(-cimag(modefreqs->data[i])*t4) * sin(creal(modefreqs->data[i])*t4));
	gsl_matrix_set(coef, 14, i, -exp(-cimag(modefreqs->data[i])*t5) * sin(creal(modefreqs->data[i])*t5));
	gsl_matrix_set(coef, 15, i, exp(-cimag(modefreqs->data[i])*t5) * 
				      (cimag(modefreqs->data[i]) * sin(creal(modefreqs->data[i])*t5)
				      -creal(modefreqs->data[i]) * cos(creal(modefreqs->data[i])*t5)));
  }
  for (i = 0; i < 2; ++i)
  {
	gsl_vector_set(hderivs, i, inspwave1->data[(i + 1) * inspwave1->vectorLength - 1]);
	gsl_vector_set(hderivs, i + nmodes, inspwave2->data[(i + 1) * inspwave2->vectorLength - 1]);
	gsl_vector_set(hderivs, i + 6, inspwave1->data[i * inspwave1->vectorLength]);
	gsl_vector_set(hderivs, i + 6 + nmodes, inspwave2->data[i * inspwave2->vectorLength]);
  }
  gsl_vector_set(hderivs, 2, inspwave1->data[4]);
  gsl_vector_set(hderivs, 2 + nmodes, inspwave2->data[4]);
  gsl_vector_set(hderivs, 3, inspwave1->data[3]);
  gsl_vector_set(hderivs, 3 + nmodes, inspwave2->data[3]);
  gsl_vector_set(hderivs, 4, inspwave1->data[2]);
  gsl_vector_set(hderivs, 4 + nmodes, inspwave2->data[2]);
  gsl_vector_set(hderivs, 5, inspwave1->data[1]);
  gsl_vector_set(hderivs, 5 + nmodes, inspwave2->data[1]);
  
  /* Complete the definition for the rest half of A */
  for (i = 0; i < nmodes; ++i)
  {
	for (k = 0; k < nmodes; ++k)
	{
	  gsl_matrix_set(coef, i, k + nmodes, - gsl_matrix_get(coef, i + nmodes, k));
	  gsl_matrix_set(coef, i + nmodes, k + nmodes, gsl_matrix_get(coef, i, k));
	}
  }

  /* Call gsl LU decomposition to solve the linear system */
  gslStatus = gsl_linalg_LU_decomp(coef, p, &s) ;
  if ( gslStatus == GSL_SUCCESS )
  {
    gslStatus = gsl_linalg_LU_solve(coef, p, hderivs, x);
  }
  if ( gslStatus != GSL_SUCCESS )
  {
    gsl_matrix_free(coef);
    gsl_vector_free(hderivs);
    gsl_vector_free(x);
    gsl_permutation_free(p);
	
    printf("Error(%d)", XLAL_EFUNC );
  }

  /* Putting solution to an XLAL vector */
  modeamps = XLALCreateREAL8Vector(2 * nmodes);

  if ( !modeamps )
  {
    gsl_matrix_free(coef);
    gsl_vector_free(hderivs);
    gsl_vector_free(x);
    gsl_permutation_free(p);
	
    printf("Error(%d)", XLAL_ENOMEM );
  }

  for (i = 0; i < nmodes; ++i)
  {
	modeamps->data[i] = gsl_vector_get(x, i);
	modeamps->data[i + nmodes] = gsl_vector_get(x, i + nmodes);
  }

  /* Free all gsl linear algebra objects */
  gsl_matrix_free(coef);
  gsl_vector_free(hderivs);
  gsl_vector_free(x);
  gsl_permutation_free(p);

  /* Build ring-down waveforms */

  REAL8 timeOffset = fmod( matchrange->data[1], dt/m) * dt;

  for (j = 0; j < rdwave1->length; ++j)
  {
	tj = j * dt - timeOffset;
	rdwave1->data[j] = 0;
	rdwave2->data[j] = 0;
	for (i = 0; i < nmodes; ++i)
	{
	  rdwave1->data[j] += exp(- tj * cimag(modefreqs->data[i]))
			* ( modeamps->data[i] * cos(tj * creal(modefreqs->data[i]))
			+   modeamps->data[i + nmodes] * sin(tj * creal(modefreqs->data[i])) );
	  rdwave2->data[j] += exp(- tj * cimag(modefreqs->data[i]))
			* (- modeamps->data[i] * sin(tj * creal(modefreqs->data[i]))
			+   modeamps->data[i + nmodes] * cos(tj * creal(modefreqs->data[i])) );
	}
  }

  XLALDestroyREAL8Vector(modeamps);
  return errcode;
}
