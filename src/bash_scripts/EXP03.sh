#!/bin/bash
set -e -o pipefail

echo "=================================================="
echo "Start"
echo "=================================================="

TASK=$1 # can be show to show the selected settings

cd ../scripts/
ROOT=../../
DATASETROOT=../../data
CONFIGBASEPATH=$ROOT/src/configs/EXP03/

SETUP=0

CONFIG_FILES=(
    00_Derm7pt-b/00_Derm7pt-b_baseline_DFR-on_color-only.ini
    00_Derm7pt-b/00_Derm7pt-b_baseline_DFR-on_color-removed.ini
    00_Derm7pt-b/00_Derm7pt-b_baseline_DFR-on_texture-only.ini
    00_Derm7pt-b/00_Derm7pt-b_baseline_DFR-on_texture-removed.ini
    00_Derm7pt-b/00_Derm7pt-b_baseline_DFR-on_shape-only.ini
    00_Derm7pt-b/00_Derm7pt-b_baseline_DFR-on_shape-removed.ini
    #
    01_Derm7pt/00_Derm7pt_baseline_DFR-on_color-only.ini
    01_Derm7pt/00_Derm7pt_baseline_DFR-on_color-removed.ini
    01_Derm7pt/00_Derm7pt_baseline_DFR-on_texture-only.ini
    01_Derm7pt/00_Derm7pt_baseline_DFR-on_texture-removed.ini
    01_Derm7pt/00_Derm7pt_baseline_DFR-on_shape-only.ini
    01_Derm7pt/00_Derm7pt_baseline_DFR-on_shape-removed.ini
    #
    02_ISICb-NV-MEL/00_ISICb-NV-MEL_baseline_DFR-on_color-only.ini
    02_ISICb-NV-MEL/00_ISICb-NV-MEL_baseline_DFR-on_color-removed.ini
    02_ISICb-NV-MEL/00_ISICb-NV-MEL_baseline_DFR-on_texture-only.ini
    02_ISICb-NV-MEL/00_ISICb-NV-MEL_baseline_DFR-on_texture-removed.ini
    02_ISICb-NV-MEL/00_ISICb-NV-MEL_baseline_DFR-on_shape-only.ini
    02_ISICb-NV-MEL/00_ISICb-NV-MEL_baseline_DFR-on_shape-removed.ini
    #
    03_ISIC/00_ISIC_baseline_DFR-on_color-only.ini
    03_ISIC/00_ISIC_baseline_DFR-on_color-removed.ini
    03_ISIC/00_ISIC_baseline_DFR-on_texture-only.ini
    03_ISIC/00_ISIC_baseline_DFR-on_texture-removed.ini
    03_ISIC/00_ISIC_baseline_DFR-on_shape-only.ini
    03_ISIC/00_ISIC_baseline_DFR-on_shape-removed.ini
)

for CID in ${!CONFIG_FILES[@]}
do
    echo "=================================================="
    echo "Setup: $CID Config: $BASEPATH${CONFIG_FILES[$CID]}" 
    echo "=================================================="

    if [[ $TASK != "show" ]]
    then
        python main_DFR.py --config $CONFIGBASEPATH${CONFIG_FILES[$CID]} -d $DATASETROOT -n 10
    fi
done

echo "=================================================="
echo "Finished execution"
echo "=================================================="
