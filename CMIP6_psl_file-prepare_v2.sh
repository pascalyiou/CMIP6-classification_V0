#!/bin/bash -l
## Prepare a few CMIP6 files for tests of classification
## Extract SLP (=psl)
## Pascal Yiou (LSCE), Oct. 2024
## Works on spiritx
## Se lance par:
## ${HOME}/programmes/RStat/CMIP6class/CMIP6_psl_file-prepare_v2.sh instna modna run yr1 yr1

module load cdo/2.0.6

## Your user name
owner=`whoami`

## Assumes that CMIP6 data have been downloaded there (OK for IPSL server)
## Tests on historical simulations
CMIPdir=/bdd/CMIP6/CMIP/

## Needs to be adjusted
OUTdir=/scratchx/${owner}/CMIP6/testML

varname=psl

##echo $1, $2, $3, $4, $5
##modname=IPSL-CM6A-LR
modname=$2
##instname=IPSL
instname=$1
scen=historical
##run=r1i1p1f1
run=$3
yr1=$4
yr2=$5

## Region à extraire
lon1=-50
lat1=30
lon2=20
lat2=65

# Création d'une grille régulière de 1° par 1° (lon-lat)
echo "gridtype = lonlat" > ${OUTdir}/grid_1x1_NA.txt
echo "xsize = 80" >> ${OUTdir}/grid_1x1_NA.txt  # 80 points pour la longitude (1° entre chaque point)
echo "ysize = 50" >> ${OUTdir}/grid_1x1_NA.txt  # 50 points pour la latitude (1° entre chaque point)
echo "xfirst = -60" >> ${OUTdir}/grid_1x1_NA.txt  # Première longitude
echo "xinc = 1" >> ${OUTdir}/grid_1x1_NA.txt  # Incrément de 1° en longitude
echo "yfirst = 20" >> ${OUTdir}/grid_1x1_NA.txt  # Première latitude
echo "yinc = 1" >> ${OUTdir}/grid_1x1_NA.txt  # Incrément de 1° en latitude

datpath=${CMIPdir}/${instname}/${modname}/${scen}/${run}/day/${varname}/*/latest

cd ${datpath}

dumfile=${varname}_${instname}_${modname}_${scen}_${run}.nc
domfile=${varname}_${instname}_${modname}_${scen}_${run}_NAtl_${yr1}-${yr2}_dom.nc
outfile=${varname}_${instname}_${modname}_${scen}_${run}_NAtl_${yr1}-${yr2}.nc

## Concatenation des fichiers
cdo cat *.nc ${OUTdir}/${dumfile}
# Interpolation des données sur la grille régulière 1x1
cdo -P ${NUM_CPUS} remapbil,${OUTdir}/grid_1x1_NA.txt ${OUTdir}/${dumfile} ${OUTdir}/${domfile}
## Extraction de la région définie par les limites de longitude et de latitude
## et de la période de yr1 à yr2
cdo sellonlatbox,${lon1},${lon2},${lat1},${lat2} -selyear,${yr1}/${yr2} ${OUTdir}/${domfile} ${OUTdir}/${outfile}

## Remove intermediate files
\rm ${OUTdir}/${dumfile} ${OUTdir}/${domfile} 


