#include "seobnre.h"

void do_waveform_generation(REAL8 **hplusData, REAL8 **hcrossData,
                            REAL8 phiRef, REAL8 deltaT, REAL8 m1, REAL8 m2, REAL8 s1z, REAL8 s2z,
                            REAL8 f_min, REAL8 e0, REAL8 distance, REAL8 inclination) {

    REAL8TimeSeries *hplus = NULL;
    REAL8TimeSeries *hcross = NULL;
    // convert solar masses and Mpc to kg and m
    REAL8 m1_kg = m1 * LAL_MSUN_SI;
    REAL8 m2_kg = m2 * LAL_MSUN_SI;
    REAL8 distance_m = distance * 1e6 * LAL_PC_SI;
    // in Panyi_elip.c
    XLALSimInspiralChooseTDWaveform(&hplus, &hcross, phiRef,
                    deltaT, m1_kg, m2_kg, 0.0, 0.0, s1z, 0.0, 0.0,
                    s2z, f_min, e0, distance_m, inclination
                    );
    *hplusData = hplus->data->data;
    *hcrossData = hcross->data->data;
}
