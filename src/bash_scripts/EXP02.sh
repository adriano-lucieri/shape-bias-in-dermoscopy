#!/bin/bash
set -e -o pipefail

echo "=================================================="
echo "Start"
echo "=================================================="

TASK=$1 # can be show to show the selected settings

cd ../scripts/
ROOT=../../
DATASETROOT=../../data
CONFIGBASEPATH=$ROOT/src/configs/EXP02/

SETUP=0

CONFIG_FILES=(
    00_Derm7pt-b/00_Derm7pt-b_baseline_on_baseline.ini
    00_Derm7pt-b/00_Derm7pt-b_baseline_on_color-only.ini
    00_Derm7pt-b/00_Derm7pt-b_baseline_on_color-removed.ini
    00_Derm7pt-b/00_Derm7pt-b_baseline_on_texture-only.ini
    00_Derm7pt-b/00_Derm7pt-b_baseline_on_texture-removed.ini
    00_Derm7pt-b/00_Derm7pt-b_baseline_on_shape-only.ini
    00_Derm7pt-b/00_Derm7pt-b_baseline_on_shape-removed.ini
    00_Derm7pt-b/01_Derm7pt-b_color-only_on_baseline.ini
    00_Derm7pt-b/01_Derm7pt-b_color-only_on_color-only.ini
    00_Derm7pt-b/01_Derm7pt-b_color-only_on_color-removed.ini
    00_Derm7pt-b/01_Derm7pt-b_color-only_on_texture-only.ini
    00_Derm7pt-b/01_Derm7pt-b_color-only_on_texture-removed.ini
    00_Derm7pt-b/01_Derm7pt-b_color-only_on_shape-only.ini
    00_Derm7pt-b/01_Derm7pt-b_color-only_on_shape-removed.ini
    00_Derm7pt-b/02_Derm7pt-b_color-removed_on_baseline.ini
    00_Derm7pt-b/02_Derm7pt-b_color-removed_on_color-only.ini
    00_Derm7pt-b/02_Derm7pt-b_color-removed_on_color-removed.ini
    00_Derm7pt-b/02_Derm7pt-b_color-removed_on_texture-only.ini
    00_Derm7pt-b/02_Derm7pt-b_color-removed_on_texture-removed.ini
    00_Derm7pt-b/02_Derm7pt-b_color-removed_on_shape-only.ini
    00_Derm7pt-b/02_Derm7pt-b_color-removed_on_shape-removed.ini
    00_Derm7pt-b/03_Derm7pt-b_texture-only_on_baseline.ini
    00_Derm7pt-b/03_Derm7pt-b_texture-only_on_color-only.ini
    00_Derm7pt-b/03_Derm7pt-b_texture-only_on_color-removed.ini
    00_Derm7pt-b/03_Derm7pt-b_texture-only_on_texture-only.ini
    00_Derm7pt-b/03_Derm7pt-b_texture-only_on_texture-removed.ini
    00_Derm7pt-b/03_Derm7pt-b_texture-only_on_shape-only.ini
    00_Derm7pt-b/03_Derm7pt-b_texture-only_on_shape-removed.ini
    00_Derm7pt-b/04_Derm7pt-b_texture-removed_on_baseline.ini
    00_Derm7pt-b/04_Derm7pt-b_texture-removed_on_color-only.ini
    00_Derm7pt-b/04_Derm7pt-b_texture-removed_on_color-removed.ini
    00_Derm7pt-b/04_Derm7pt-b_texture-removed_on_texture-only.ini
    00_Derm7pt-b/04_Derm7pt-b_texture-removed_on_texture-removed.ini
    00_Derm7pt-b/04_Derm7pt-b_texture-removed_on_shape-only.ini
    00_Derm7pt-b/04_Derm7pt-b_texture-removed_on_shape-removed.ini
    00_Derm7pt-b/05_Derm7pt-b_shape-only_on_baseline.ini
    00_Derm7pt-b/05_Derm7pt-b_shape-only_on_color-only.ini
    00_Derm7pt-b/05_Derm7pt-b_shape-only_on_color-removed.ini
    00_Derm7pt-b/05_Derm7pt-b_shape-only_on_texture-only.ini
    00_Derm7pt-b/05_Derm7pt-b_shape-only_on_texture-removed.ini
    00_Derm7pt-b/05_Derm7pt-b_shape-only_on_shape-only.ini
    00_Derm7pt-b/05_Derm7pt-b_shape-only_on_shape-removed.ini
    00_Derm7pt-b/06_Derm7pt-b_shape-removed_on_baseline.ini
    00_Derm7pt-b/06_Derm7pt-b_shape-removed_on_color-only.ini
    00_Derm7pt-b/06_Derm7pt-b_shape-removed_on_color-removed.ini
    00_Derm7pt-b/06_Derm7pt-b_shape-removed_on_texture-only.ini
    00_Derm7pt-b/06_Derm7pt-b_shape-removed_on_texture-removed.ini
    00_Derm7pt-b/06_Derm7pt-b_shape-removed_on_shape-only.ini
    00_Derm7pt-b/06_Derm7pt-b_shape-removed_on_shape-removed.ini
    #
    01_Derm7pt/00_Derm7pt_baseline_on_baseline.ini
    01_Derm7pt/00_Derm7pt_baseline_on_color-only.ini
    01_Derm7pt/00_Derm7pt_baseline_on_color-removed.ini
    01_Derm7pt/00_Derm7pt_baseline_on_texture-only.ini
    01_Derm7pt/00_Derm7pt_baseline_on_texture-removed.ini
    01_Derm7pt/00_Derm7pt_baseline_on_shape-only.ini
    01_Derm7pt/00_Derm7pt_baseline_on_shape-removed.ini
    01_Derm7pt/01_Derm7pt_color-only_on_baseline.ini
    01_Derm7pt/01_Derm7pt_color-only_on_color-only.ini
    01_Derm7pt/01_Derm7pt_color-only_on_color-removed.ini
    01_Derm7pt/01_Derm7pt_color-only_on_texture-only.ini
    01_Derm7pt/01_Derm7pt_color-only_on_texture-removed.ini
    01_Derm7pt/01_Derm7pt_color-only_on_shape-only.ini
    01_Derm7pt/01_Derm7pt_color-only_on_shape-removed.ini
    01_Derm7pt/02_Derm7pt_color-removed_on_baseline.ini
    01_Derm7pt/02_Derm7pt_color-removed_on_color-only.ini
    01_Derm7pt/02_Derm7pt_color-removed_on_color-removed.ini
    01_Derm7pt/02_Derm7pt_color-removed_on_texture-only.ini
    01_Derm7pt/02_Derm7pt_color-removed_on_texture-removed.ini
    01_Derm7pt/02_Derm7pt_color-removed_on_shape-only.ini
    01_Derm7pt/02_Derm7pt_color-removed_on_shape-removed.ini
    01_Derm7pt/03_Derm7pt_texture-only_on_baseline.ini
    01_Derm7pt/03_Derm7pt_texture-only_on_color-only.ini
    01_Derm7pt/03_Derm7pt_texture-only_on_color-removed.ini
    01_Derm7pt/03_Derm7pt_texture-only_on_texture-only.ini
    01_Derm7pt/03_Derm7pt_texture-only_on_texture-removed.ini
    01_Derm7pt/03_Derm7pt_texture-only_on_shape-only.ini
    01_Derm7pt/03_Derm7pt_texture-only_on_shape-removed.ini
    01_Derm7pt/04_Derm7pt_texture-removed_on_baseline.ini
    01_Derm7pt/04_Derm7pt_texture-removed_on_color-only.ini
    01_Derm7pt/04_Derm7pt_texture-removed_on_color-removed.ini
    01_Derm7pt/04_Derm7pt_texture-removed_on_texture-only.ini
    01_Derm7pt/04_Derm7pt_texture-removed_on_texture-removed.ini
    01_Derm7pt/04_Derm7pt_texture-removed_on_shape-only.ini
    01_Derm7pt/04_Derm7pt_texture-removed_on_shape-removed.ini
    01_Derm7pt/05_Derm7pt_shape-only_on_baseline.ini
    01_Derm7pt/05_Derm7pt_shape-only_on_color-only.ini
    01_Derm7pt/05_Derm7pt_shape-only_on_color-removed.ini
    01_Derm7pt/05_Derm7pt_shape-only_on_texture-only.ini
    01_Derm7pt/05_Derm7pt_shape-only_on_texture-removed.ini
    01_Derm7pt/05_Derm7pt_shape-only_on_shape-only.ini
    01_Derm7pt/05_Derm7pt_shape-only_on_shape-removed.ini
    01_Derm7pt/06_Derm7pt_shape-removed_on_baseline.ini
    01_Derm7pt/06_Derm7pt_shape-removed_on_color-only.ini
    01_Derm7pt/06_Derm7pt_shape-removed_on_color-removed.ini
    01_Derm7pt/06_Derm7pt_shape-removed_on_texture-only.ini
    01_Derm7pt/06_Derm7pt_shape-removed_on_texture-removed.ini
    01_Derm7pt/06_Derm7pt_shape-removed_on_shape-only.ini
    01_Derm7pt/06_Derm7pt_shape-removed_on_shape-removed.ini
    #
    02_ISICb-NV-MEL/00_ISICb-NV-MEL_baseline_on_baseline.ini
    02_ISICb-NV-MEL/00_ISICb-NV-MEL_baseline_on_color-only.ini
    02_ISICb-NV-MEL/00_ISICb-NV-MEL_baseline_on_color-removed.ini
    02_ISICb-NV-MEL/00_ISICb-NV-MEL_baseline_on_texture-only.ini
    02_ISICb-NV-MEL/00_ISICb-NV-MEL_baseline_on_texture-removed.ini
    02_ISICb-NV-MEL/00_ISICb-NV-MEL_baseline_on_shape-only.ini
    02_ISICb-NV-MEL/00_ISICb-NV-MEL_baseline_on_shape-removed.ini
    02_ISICb-NV-MEL/01_ISICb-NV-MEL_color-only_on_baseline.ini
    02_ISICb-NV-MEL/01_ISICb-NV-MEL_color-only_on_color-only.ini
    02_ISICb-NV-MEL/01_ISICb-NV-MEL_color-only_on_color-removed.ini
    02_ISICb-NV-MEL/01_ISICb-NV-MEL_color-only_on_texture-only.ini
    02_ISICb-NV-MEL/01_ISICb-NV-MEL_color-only_on_texture-removed.ini
    02_ISICb-NV-MEL/01_ISICb-NV-MEL_color-only_on_shape-only.ini
    02_ISICb-NV-MEL/01_ISICb-NV-MEL_color-only_on_shape-removed.ini
    02_ISICb-NV-MEL/02_ISICb-NV-MEL_color-removed_on_baseline.ini
    02_ISICb-NV-MEL/02_ISICb-NV-MEL_color-removed_on_color-only.ini
    02_ISICb-NV-MEL/02_ISICb-NV-MEL_color-removed_on_color-removed.ini
    02_ISICb-NV-MEL/02_ISICb-NV-MEL_color-removed_on_texture-only.ini
    02_ISICb-NV-MEL/02_ISICb-NV-MEL_color-removed_on_texture-removed.ini
    02_ISICb-NV-MEL/02_ISICb-NV-MEL_color-removed_on_shape-only.ini
    02_ISICb-NV-MEL/02_ISICb-NV-MEL_color-removed_on_shape-removed.ini
    02_ISICb-NV-MEL/03_ISICb-NV-MEL_texture-only_on_baseline.ini
    02_ISICb-NV-MEL/03_ISICb-NV-MEL_texture-only_on_color-only.ini
    02_ISICb-NV-MEL/03_ISICb-NV-MEL_texture-only_on_color-removed.ini
    02_ISICb-NV-MEL/03_ISICb-NV-MEL_texture-only_on_texture-only.ini
    02_ISICb-NV-MEL/03_ISICb-NV-MEL_texture-only_on_texture-removed.ini
    02_ISICb-NV-MEL/03_ISICb-NV-MEL_texture-only_on_shape-only.ini
    02_ISICb-NV-MEL/03_ISICb-NV-MEL_texture-only_on_shape-removed.ini
    02_ISICb-NV-MEL/04_ISICb-NV-MEL_texture-removed_on_baseline.ini
    02_ISICb-NV-MEL/04_ISICb-NV-MEL_texture-removed_on_color-only.ini
    02_ISICb-NV-MEL/04_ISICb-NV-MEL_texture-removed_on_color-removed.ini
    02_ISICb-NV-MEL/04_ISICb-NV-MEL_texture-removed_on_texture-only.ini
    02_ISICb-NV-MEL/04_ISICb-NV-MEL_texture-removed_on_texture-removed.ini
    02_ISICb-NV-MEL/04_ISICb-NV-MEL_texture-removed_on_shape-only.ini
    02_ISICb-NV-MEL/04_ISICb-NV-MEL_texture-removed_on_shape-removed.ini
    02_ISICb-NV-MEL/05_ISICb-NV-MEL_shape-only_on_baseline.ini
    02_ISICb-NV-MEL/05_ISICb-NV-MEL_shape-only_on_color-only.ini
    02_ISICb-NV-MEL/05_ISICb-NV-MEL_shape-only_on_color-removed.ini
    02_ISICb-NV-MEL/05_ISICb-NV-MEL_shape-only_on_texture-only.ini
    02_ISICb-NV-MEL/05_ISICb-NV-MEL_shape-only_on_texture-removed.ini
    02_ISICb-NV-MEL/05_ISICb-NV-MEL_shape-only_on_shape-only.ini
    02_ISICb-NV-MEL/05_ISICb-NV-MEL_shape-only_on_shape-removed.ini
    02_ISICb-NV-MEL/06_ISICb-NV-MEL_shape-removed_on_baseline.ini
    02_ISICb-NV-MEL/06_ISICb-NV-MEL_shape-removed_on_color-only.ini
    02_ISICb-NV-MEL/06_ISICb-NV-MEL_shape-removed_on_color-removed.ini
    02_ISICb-NV-MEL/06_ISICb-NV-MEL_shape-removed_on_texture-only.ini
    02_ISICb-NV-MEL/06_ISICb-NV-MEL_shape-removed_on_texture-removed.ini
    02_ISICb-NV-MEL/06_ISICb-NV-MEL_shape-removed_on_shape-only.ini
    02_ISICb-NV-MEL/06_ISICb-NV-MEL_shape-removed_on_shape-removed.ini
    #
    03_ISIC/00_ISIC_baseline_on_baseline.ini
    03_ISIC/00_ISIC_baseline_on_color-only.ini
    03_ISIC/00_ISIC_baseline_on_color-removed.ini
    03_ISIC/00_ISIC_baseline_on_texture-only.ini
    03_ISIC/00_ISIC_baseline_on_texture-removed.ini
    03_ISIC/00_ISIC_baseline_on_shape-only.ini
    03_ISIC/00_ISIC_baseline_on_shape-removed.ini
    03_ISIC/01_ISIC_color-only_on_baseline.ini
    03_ISIC/01_ISIC_color-only_on_color-only.ini
    03_ISIC/01_ISIC_color-only_on_color-removed.ini
    03_ISIC/01_ISIC_color-only_on_texture-only.ini
    03_ISIC/01_ISIC_color-only_on_texture-removed.ini
    03_ISIC/01_ISIC_color-only_on_shape-only.ini
    03_ISIC/01_ISIC_color-only_on_shape-removed.ini
    03_ISIC/02_ISIC_color-removed_on_baseline.ini
    03_ISIC/02_ISIC_color-removed_on_color-only.ini
    03_ISIC/02_ISIC_color-removed_on_color-removed.ini
    03_ISIC/02_ISIC_color-removed_on_texture-only.ini
    03_ISIC/02_ISIC_color-removed_on_texture-removed.ini
    03_ISIC/02_ISIC_color-removed_on_shape-only.ini
    03_ISIC/02_ISIC_color-removed_on_shape-removed.ini
    03_ISIC/03_ISIC_texture-only_on_baseline.ini
    03_ISIC/03_ISIC_texture-only_on_color-only.ini
    03_ISIC/03_ISIC_texture-only_on_color-removed.ini
    03_ISIC/03_ISIC_texture-only_on_texture-only.ini
    03_ISIC/03_ISIC_texture-only_on_texture-removed.ini
    03_ISIC/03_ISIC_texture-only_on_shape-only.ini
    03_ISIC/03_ISIC_texture-only_on_shape-removed.ini
    03_ISIC/04_ISIC_texture-removed_on_baseline.ini
    03_ISIC/04_ISIC_texture-removed_on_color-only.ini
    03_ISIC/04_ISIC_texture-removed_on_color-removed.ini
    03_ISIC/04_ISIC_texture-removed_on_texture-only.ini
    03_ISIC/04_ISIC_texture-removed_on_texture-removed.ini
    03_ISIC/04_ISIC_texture-removed_on_shape-only.ini
    03_ISIC/04_ISIC_texture-removed_on_shape-removed.ini
    03_ISIC/05_ISIC_shape-only_on_baseline.ini
    03_ISIC/05_ISIC_shape-only_on_color-only.ini
    03_ISIC/05_ISIC_shape-only_on_color-removed.ini
    03_ISIC/05_ISIC_shape-only_on_texture-only.ini
    03_ISIC/05_ISIC_shape-only_on_texture-removed.ini
    03_ISIC/05_ISIC_shape-only_on_shape-only.ini
    03_ISIC/05_ISIC_shape-only_on_shape-removed.ini
    03_ISIC/06_ISIC_shape-removed_on_baseline.ini
    03_ISIC/06_ISIC_shape-removed_on_color-only.ini
    03_ISIC/06_ISIC_shape-removed_on_color-removed.ini
    03_ISIC/06_ISIC_shape-removed_on_texture-only.ini
    03_ISIC/06_ISIC_shape-removed_on_texture-removed.ini
    03_ISIC/06_ISIC_shape-removed_on_shape-only.ini
    03_ISIC/06_ISIC_shape-removed_on_shape-removed.ini
)

for CID in ${!CONFIG_FILES[@]}
do
    echo "=================================================="
    echo "Setup: $CID Config: $BASEPATH${CONFIG_FILES[$CID]}" 
    echo "=================================================="

    if [[ $TASK != "show" ]]
    then
        python main.py --config $CONFIGBASEPATH${CONFIG_FILES[$CID]} -d $DATASETROOT -n 10
    fi
done

echo "=================================================="
echo "Finished execution"
echo "=================================================="
