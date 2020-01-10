"""

A file containing file system-specific information relating to GWTC-1 for the pycentricity package.

"""
psd_files_base_path = '/home/isobel.romero-shaw/lvc_pe_sample_release/'

event_psd_file_path = dict(
    GW150914=dict(
        H1=psd_files_base_path + "O1/PE/GW150914/rerun_O2_catalog/h1_psd.dat",
        L1=psd_files_base_path + "O1/PE/GW150914/rerun_O2_catalog/l1_psd.dat"
    ),
    GW151012=dict(
        H1=psd_files_base_path + "O1/PE/LVT151012/rerun_O2_catalog/BayesWave_median_PSD_H1.dat",
        L1=psd_files_base_path + "O1/PE/LVT151012/rerun_O2_catalog/BayesWave_median_PSD_L1.dat"
    ),
    GW151226=dict(
        H1=psd_files_base_path + "O1/PE/GW151226/rerun_O2_catalog/BayesWave_median_PSD_H1.dat",
        L1=psd_files_base_path + "O1/PE/GW151226/rerun_O2_catalog/BayesWave_median_PSD_L1.dat"
    ),
    GW170104=dict(
        H1=psd_files_base_path + "O2/PE/GW170104/rerun_O2_catalog/BayesWave_median_PSD_H1.dat",
        L1=psd_files_base_path + "O2/PE/GW170104/rerun_O2_catalog/BayesWave_median_PSD_L1.dat"
    ),
    GW170608=dict(
        H1=psd_files_base_path + "/O2/PE/GW170608/GW170608_C02_reruns/h1_psd_C02.dat",
        L1=psd_files_base_path + "O2/PE/GW170608/GW170608_C02_reruns/l1_psd_C02.dat"
    ),
    GW170729=dict(
        H1=psd_files_base_path + "O2/PE/GW170729/Median_PSD_H1.dat",
        L1=psd_files_base_path + "O2/PE/GW170729/Median_PSD_L1.dat"
    ),
    GW170809=dict(
        H1=psd_files_base_path + "/home/isobel.romero-shaw/public_html/known_events/GW170809/dynesty_isobel_defaults/GW170809_LIGO_Hanford_PSD1Hz_psd.txt",
        L1=psd_files_base_path + "/home/isobel.romero-shaw/public_html/known_events/GW170809/dynesty_isobel_defaults/GW170809_LIGO_Livingston_PSD1Hz_psd.txt"
    ),
    GW170814=dict(
        H1=psd_files_base_path + "O2/PE/GW170814/BayesWave_median_PSD_H1.dat",
        L1=psd_files_base_path + "O2/PE/GW170814/BayesWave_median_PSD_L1.dat",
        V1=psd_files_base_path + "O2/PE/GW170814/BayesWave_median_PSD_V1.dat",
    ),
    GW170818=dict(
        H1=psd_files_base_path + "O2/PE/GW170818/psd/BayesWave_median_PSD_H1.dat",
        L1=psd_files_base_path + "O2/PE/GW170818/psd/BayesWave_median_PSD_L1.dat",
        V1=psd_files_base_path + "O2/PE/GW170818/psd/BayesWave_median_PSD_V1.dat",
    ),
    GW170823=dict(
        H1=psd_files_base_path + "O2/PE/GW170823/BayesWave_median_PSD_H1.dat",
        L1=psd_files_base_path + "O2/PE/GW170823/BayesWave_median_PSD_L1.dat"
    ),
)
