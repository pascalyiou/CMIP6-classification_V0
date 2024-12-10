## Need to: module load R/4.4.1 tensorflow/2.15.0
## Can we identify a CMIP6 model from one SLP map over the eastern North
## Atlantic?
## Randomisation des modeles
## Verification de la classification à partir d'un autre jeu de données
## (autres runs CMIP6)
## Pascal Yiou (LSCE),  Nov. 2024
## Demande d'avoir lancé:
## sbatch ${HOME}/programmes/RStat/CMIP6class/CMIP6_classif-v1.sh NMOD
## Fonctionne sur la machine GPU hal de l'IPSL
## Se lance par:
## R CMD BATCH "--args SAISON NMOD JOBID" ${HOME}/programmes/RStat/CMIP6class/CMIP6_tensorflow-verif_v1.R
## ou JOBID=${SLURM_JOB_ID}
## est décrit dans le script qui a lance le programme précédent

SI=Sys.info()
user=SI[["user"]]
if(SI[[1]] == "Linux"){
    Rsource=paste("/home/",user,"/programmes/RStat/",sep="")
    DATdir=paste("/scratchx/",user,"/CMIP6/testML",sep="")
    ERAdir=paste("/scratchx/",user,"/ERA5/",sep="")
    OUTdir = paste("/scratchx/",user,"/CMIP6/testML",sep="") ## Needs to be adapted
    Sys.setenv('TAR'='/usr/bin/tar') # Sinon R ne trouve pas le executable tar
    .libPaths(c("/home/pyiou/R/x86_64-pc-linux-gnu-library/4.0", .libPaths()) ) # sinon R met en priorité un libpath sur lequel l'utilisateur n'a pas les droits d'écriture. chemin à modifier avec son propre libpath. 

    reticulate::use_condaenv('/net/nfs/tools/u20/Python/miniconda3_py311_23.11.0-2/envs/tensorflow-2.15.0/bin/python') # pour spécifier d'utiliser le python du module tensorflow/2.15.0
}
if(SI[[1]] == "Darwin"){
    Rsource=paste("/Users/",user,"/programmes/RStat/",sep="")
    DATdir = paste("/Users/",user,"/work/CMIP6_IA/",sep="") ## Needs to be adapted
    OUTdir=paste("/Users/",user,"/work/CMIP6_IA/",sep="")  ## Needs to be adapted
}

## Code set up
## Pour l'IA
library(keras3)
nneurons=256 ## Nombre de neuronnes. Par défaut: nneurons=128
## Pour lire les fichier netcdf
library(ncdf4)
library(ncdf4.helpers)
## Pour concaténer des tableaux en 3d
library(abind)
## Pour tracer des cartes de contrôle
library(fields)
library(scales)
source(paste(Rsource,"imagecont.R",sep=""))

## Arguments du programme
args=(commandArgs(TRUE))
print(args)
i=1
if(length(args)>0){
    seas = args[i];i=i+1 ## Saison pour le calcul
    nmod = as.numeric(args[i]) ;i=i+1 ## Nombre de modeles à prendre
    jobid = args[i] ;i=i+1 ## Indice du job
}else{
    seas = "JJA" ## Saison pour le calcul
    nmod = 9 ## Nombre de modeles à prendre
    jobid = "16722"
}

## Lecture des résultats de l'étape précédente
##setwd("/home/pyiou/RESULTS")
##setwd(paste("/home/",user,"/RESULTS",sep=""))
setwd(paste("/net/nfs/ssd1/",user,"/IA_RESULTS",sep=""))
##filout=paste("test_CMIP-class_v1_",nmod,"_",seas,"_",jobid,sep="")
fname=paste("test_CMIP-class_v1_",nmod,"_",seas,"_",jobid,sep="")
## save(file=paste(filout,".Rdat",sep=""),l.OK,l.names,seas,l.mod.sel)
load(paste(fname,".Rdat",sep=""))

nclass=nmod+1

## Chargement du modèle CNN
nname=paste("SLP_CMIP6_CNN-",nneurons,"-",nclass,"-model-",seas,"_",jobid,sep="")
model=load_model(paste(nname,".keras",sep=""))

## Date set up
## Liste des saisons
l.seas=list(DJF=c(12,1,2),MAM=c(3,4,5),JJA=c(6,7,8),SON=c(9,10,11))

## Vecteur avec les noms de simulations
Lnames=c()
## Vecteur avec les dates
Ltimes=c()

## Read ERA5 reanalysis
setwd(ERAdir)
nc=nc_open("era5_msl_daily_NAtl_1970-2000.nc")
time.dum = ncdf4.helpers::nc.get.time.series(nc)
time.ERA=as.numeric(format(time.dum, "%Y%m%d"))
mm=floor(time.ERA/100) %% 100
SLP=ncvar_get(nc,"msl")/100
## Normalisation par la moyenne générale
SLP=SLP-mean(SLP)
## Les longitudes et latitudes sont les mêmes pour tout le monde
lon=ncvar_get(nc,"lon")
lat=ncvar_get(nc,"lat")
nc_close(nc)
SLP=aperm(SLP,c(3,1,2))
Iseas=which(mm %in% l.seas[[seas]])
SLP=SLP[Iseas,,]
## Vecteurs avec le nom de la reanalyse ou des modeles
Lnames=rep("ECMWF_ERA5",times=dim(SLP)[1])
## Vecteur avec le temps
Ltimes=time.ERA[Iseas]
l.mod="ECMWF_ERA5"
Imodel=rep(1,times=length(Ltimes))

setwd(DATdir)
## Liste des fichiers CMIP6

## Read CMIP6 simulation data
## l.mod.sel=c("BCC_BCC-ESM1","CAS_FGOALS-g3","CCCma_CanESM5",
##                   "CNRM-CERFACS_CNRM-CM6-1","CSIRO_ACCESS-ESM1-5",
##                   "EC-Earth-Consortium_EC-Earth3",
##  ##                 "HAMMOZ-Consortium_MPI-ESM-1-2-HAM",
##                   "INM_INM-CM5-0","IPSL_IPSL-CM6A-LR","MIROC_MIROC6",
##                   "MOHC_UKESM1-0-LL","MPI-M_MPI-ESM1-2-LR","MRI_MRI-ESM2-0",
##                   "NCAR_CESM2","NCC_NorCPM1","NIMS-KMA_KACE-1-0-G",
##             "NUIST_NESM3")
imod=2
for(mod in l.mod.sel){
    ls.mod=system(paste("ls psl_",mod,"_historical_r?i?p*_1970-2000.nc",sep=""),
                  intern=TRUE)
    firun=ls.mod[3] ## On prend le second
    print(paste("Read ",firun))
    nc=nc_open(firun)
    time.dum = ncdf4.helpers::nc.get.time.series(nc)
    time.CMIP=as.numeric(format(time.dum, "%Y%m%d"))
    mm=floor(time.CMIP/100) %% 100
    SLPdum=ncvar_get(nc,"psl")/100
    nc_close(nc)
    SLPdum=aperm(SLPdum,c(3,1,2))
## Normalisation par la moyenne générale
    SLPdum=SLPdum-mean(SLPdum)
## Extraction de la saison seas    
    Iseas=which(mm %in% l.seas[[seas]])
    SLPdum=SLPdum[Iseas,,]
    Ltimes=c(Ltimes,time.CMIP[Iseas])
    Lnames=c(Lnames,rep(mod,times=dim(SLPdum)[1]))
    Imodel=c(Imodel,rep(imod,times=dim(SLPdum)[1]))
    SLP=abind(SLP,SLPdum,along=1)
    l.mod=c(l.mod,mod)
    rm(SLPdum)
    imod=imod+1
}## end for mod

## Score du modele
score <- model %>% evaluate(SLP, Imodel, verbose = 0)

## Prediction sur les modeles
predictions <- model %>% predict(SLP)
I.pred=apply(predictions,1,which.max)-1

## Score par modele
OK.rate=c()
for(imod in 1:length(l.mod)){
    OK.imod=length(which(I.pred[Imodel==imod] == imod))/length(which(Imodel==imod))
    OK.rate=c(OK.rate,OK.imod)
}

## Composites de SLP en cas de confusion de classif, pour chaque modèle
SLP.meandis=c()
SLP.sddis=c()
for(imod in 1:length(l.mod)){
    SLPmean=apply(SLP[Imodel==imod & I.pred != Imodel,,],c(2,3),mean)
    SLPsd=apply(SLP[Imodel==imod & I.pred != Imodel,,],c(2,3),sd)
    SLP.meandis=abind(SLP.meandis, SLPmean,along=3)
    SLP.sddis=abind(SLP.sddis, SLPsd,along=3)
}

SLPERA5mean=apply(SLP[Imodel==1,,],c(2,3),mean)

## A qui sont attribuées les SLP?
Apred=c()
for(i in 1:length(l.mod)){
    A=apply(predictions[Imodel==i,2:ncol(predictions)],2,mean)
    Apred=rbind(Apred,A)
}

setwd(paste("/home/",user,"/RESULTS",sep=""))
fname=paste("proba-class-v1_CMIP6_",seas,"_",nmod,sep="")

## Figures
pdf(file=paste(fname,".pdf",sep=""),width=10)
par(mar=c(8,8,1,1))
zlim=round(range(Apred),digits=1)
par(mar=c(9,9,1,2))
image.plot(1:nrow(Apred),1:ncol(Apred),t(Apred),axes=FALSE,xlab="",ylab="",
           zlim=zlim,nlevel=10,col=rev(pal_grey(0, 1)(10)))
axis(side=2,at=1:ncol(Apred),labels=l.names,las=2)
axis(side=1,at=1:nrow(Apred),labels=l.names,las=2)
box()
dev.off()

##
pdf(file=paste("map-diff_v1_",seas,"_",nmod,".pdf",sep=""))
imod=10
zlev=seq(-5,5,by=1)
layout(matrix(1:18,6,3))
for(imod in 1:length(l.mod)){
    image.cont(lon,lat,SLP.meandis[,,imod]-SLPERA5mean,transpose=FALSE,
               mar=c(2,3,1,3),xlab="",ylab="",zlev=zlev,satur=TRUE)
    image.cont.c(lon,lat,SLP.sddis[,,imod],transpose=FALSE,
                 mar=c(2,3,1,3),xlab="",ylab="",add=TRUE)
legend("bottomright",l.names[imod])
legend("topleft",paste("(",letters[imod],")",sep=""),bty="n")
}
dev.off()

q("no")

