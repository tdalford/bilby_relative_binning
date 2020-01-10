#include <Python.h>
#include <stdlib.h>
#include "Panyidatatypes.h"
#include "PanyiLALConstants.h"
#include "Panyi_elip.h"

typedef struct gravitationalWaveStrain {
    REAL8TimeSeries * hplus, * hcross;
}
Strain;

void do_waveform_generation(REAL8 **hplusData, REAL8 **hcrossData, REAL8 phiRef, REAL8 deltaT, REAL8 m1, REAL8 m2, REAL8 s1z, REAL8 s2z,
                            REAL8 f_min, REAL8 e0, REAL8 distance, REAL8 inclination);

//PyObject * aligned_spin_waveform(PyObject * self, PyObject * args);

