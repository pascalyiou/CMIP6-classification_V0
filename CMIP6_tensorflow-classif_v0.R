## Attention: v0 de référence. NE PLUS TOUCHER!
## Run a tensorflow example test
## Install and prepare tensor flow on hal
##
## Need to: module load R/4.4.1 tensorflow/2.15.0
##
## Can we identify a CMIP6 model from one SLP map over the eastern North
## Atlantic?
## Pascal Yiou (LSCE), Oct. 2024, Nov. 2024
## Version v0 proche de l'originale.
## C'est celle qui marche le mieux. NE PAS TOUCHER!
## Fonctionne sur la machine GPU hal de l'IPSL
## Se lance par:
## R CMD BATCH "--args SEAS NMOD" ${HOME}/programmes/RStat/CMIP6class/V0/CMIP6_tensorflow-classif_v0.R 

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
## Pour lire les fichier netcdf
library(ncdf4)
library(ncdf4.helpers)
## Pour concaténer des tableaux en 3d
library(abind)
## Pour tracer des cartes de contrôle
source(paste(Rsource,"imagecont.R",sep=""))

## Arguments du programme
args=(commandArgs(TRUE))
print(args)
i=1
if(length(args)>0){
    seas = args[i];i=i+1 ## Saison pour le calcul
    nmod = as.numeric(args[i]) ;i=i+1 ## Nombre de modeles à prendre
}else{ ## S'il n'y a pas d'argument
    seas = "JJA" ## Saison pour le calcul
    nmod = 16 ## Nombre de modeles à prendre
}

nneurons=256

## Date set up
## Liste des saisons
l.seas=list(DJF=c(12,1,2),MAM=c(3,4,5),JJA=c(6,7,8),SON=c(9,10,11))
## On va repeter l'apprentissage nsim fois
nsim=20
## Nombre d'échantillons d'entraînement pour chaque modèle
ntrain=2000

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

setwd(DATdir)
## Liste des fichiers CMIP6
ls.fi=system("ls psl*1970-2000.nc",intern=TRUE)
dum=strsplit(ls.fi,"_historical_")
adum=t(matrix(unlist(dum),nrow=2))
## Nom des groupes et modèles
dum=t(matrix(unlist(strsplit(adum[,1],"psl_")),nrow=2))
namsim=dum[,2]
## Nom du run du modèle
namrun=unlist(strsplit(adum[,2],"_NAtl_1970-2000.nc"))
## Comptage des simulations par modèle
unamsim=unique(namsim)
countsim= tapply(namsim,namsim,length)
## Liste des modèles avec plus de 2 runs
l.mod2=unamsim[which(countsim >= 2)]

## Read CMIP6 simulation data
l.mod.sel=c("BCC_BCC-ESM1","CAS_FGOALS-g3","CCCma_CanESM5","CNRM-CERFACS_CNRM-ESM2-1","IPSL_IPSL-CM6A-LR")
l.mod.sel=c("BCC_BCC-ESM1","CAS_FGOALS-g3","CCCma_CanESM5",
                  "CNRM-CERFACS_CNRM-CM6-1","CSIRO_ACCESS-ESM1-5",
                  "EC-Earth-Consortium_EC-Earth3",
 ##                 "HAMMOZ-Consortium_MPI-ESM-1-2-HAM",
                  "INM_INM-CM5-0","IPSL_IPSL-CM6A-LR","MIROC_MIROC6",
            "MOHC_HadGEM3-GC31-LL",
##            "MOHC_UKESM1-0-LL",
            "MPI-M_MPI-ESM1-2-LR","MRI_MRI-ESM2-0",
                  "NCAR_CESM2","NCC_NorCPM1","NIMS-KMA_KACE-1-0-G",
            "NUIST_NESM3")
## On considère les nmod premiers modèles dans l'ordre alphabétique
l.mod.sel=l.mod.sel[1:nmod]
for(mod in l.mod.sel){
    ls.mod=system(paste("ls psl_",mod,"_historical_r?i?p*_1970-2000.nc",sep=""),
                  intern=TRUE)
    firun=ls.mod[1] ## On prend le premier
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
    SLP=abind(SLP,SLPdum,along=1) ## Concaténation
    l.mod=c(l.mod,mod)
    rm(SLPdum)
}## end for mod

## Calcul des mois de chaque date
mm=floor(Ltimes/100) %% 100
## l.modname=unlist(strsplit(l.mod,"_"))[seq(2,2*length(l.mod),by=2)]

## Transformer les noms de modèles en chiffres de 1 à nmod
I.mod=match(Lnames,l.mod)

## Classification par modèle pour chaque saison
L.OK.seas=list()
L.OK.seas.test=list()
## Selection d'une saison
##I.seas=which(mm %in% l.seas[[seas]])
I.seas=1:length(Ltimes)
## Indices par modele dans SLP et tas pour la saison seas
l.i=c()
for(i in unique(I.mod)){
    l.i=c(l.i, min(which(I.mod[I.seas]==i)))
}

## Indices de l'ensemble d'entrainement
itrain=c()
for(i in unique(I.mod)){
    itrain=c(itrain, c(l.i[i]:(l.i[i]+ntrain)))
}
## Indices de l'ensemble de verification
itest=c()
for(i in 1:(length(unique(I.mod))-1)){
    itest=c(itest, c((l.i[i]+ntrain+1):(l.i[i+1]-1)))
}
itest=c(itest, c((l.i[i+1]+ntrain+1):length(I.mod[I.seas])))

## Definition des ensembles d'apprentissage et de verification
trainSLP=SLP[I.seas[itrain],,]
trainMOD=I.mod[I.seas[itrain]]
veriSLP=SLP[I.seas[itest],,]
veriMOD=I.mod[I.seas[itest]]

## Adapté de l'exemple dans:
## https://tensorflow.rstudio.com/tutorials/keras/classification
## fashion_mnist <- dataset_fashion_mnist()

## c(train_images, train_labels) %<-% fashion_mnist$train
## c(test_images, test_labels) %<-% fashion_mnist$test

## Normalisation des données
## train_images <- train_images / 255
## test_images <- test_images / 255
## Nom generique du modèle de NN
nclass=nmod+1
nname=paste("SLP_CMIP6_CNN-",nneurons,"-",nclass,"-model-",
            seas,"_v0",sep="")
setwd(paste("/home/",user,"/RESULTS",sep=""))

## Classification par de Monte Carlo
l.OK=c()
l.OK.test=c()
score.acc=0
for(i in 1:nsim){
## Definition du modèle de réseau de neuronnes
    nx=dim(trainSLP)[2]
    ny=dim(trainSLP)[3]
    nclass=length(unique(l.mod))
    model <- keras_model_sequential()
## Changement par rapport à l'exemple sur les vêtements:
## on prend une couche de 256 neuronnes
    model %>%
        layer_flatten(input_shape = c(nx, ny)) %>%
        layer_dense(units = nneurons, activation = 'relu') %>%
        layer_dense(units = nclass+1, activation = 'softmax')

## Compilation du modèle
    model %>% compile(
                  optimizer = 'adam',
                  loss = 'sparse_categorical_crossentropy',
                  metrics = c('accuracy')
              )

    ## Entrainement du modele sur les données d'entrainement
    model %>% fit(trainSLP, trainMOD, epochs = 5, verbose = 2)

    ## Score du modele
    score <- model %>% evaluate(veriSLP, veriMOD, verbose = 0)
## Sauvegarder le modèle complet (architecture + poids) s'il est meilleur
## que le précédent
    if(score[["accuracy"]] > score.acc){
        score.acc=score[["accuracy"]]
        model %>% save_model(paste(nname,".keras",sep=""),overwrite=TRUE)
    }

    ## Prediction sur l'ensemble de verification
    predictions <- model %>% predict(veriSLP)

    I.pred=apply(predictions,1,which.max)-1

    ## Taux de succes pour chaque modele sur la periode de verification
    OK.rate=c()
    for(imod in 1:length(l.mod)){
        OK.imod=length(which(I.pred[veriMOD==imod] == imod))/length(which(veriMOD==imod))
        OK.rate=c(OK.rate,OK.imod)
    }
    l.OK=rbind(l.OK,OK.rate)
}## end for i 

l.names=matrix(unlist(strsplit(l.mod,"_")),nrow=2)[2,]

## Sauvegarde du résultat
setwd(paste("/home/",user,"/RESULTS",sep=""))
filout=paste("test_CMIP-class_v0_",nmod,"_",seas,sep="")
save(file=paste(filout,".Rdat",sep=""),l.OK,l.names,l.mod.sel,seas,predictions)

## Figure de score avec boxplots
pdf(paste(filout,".pdf",sep=""),width=8)
i=1
par(mar=c(9,4,1,1))
boxplot(l.OK,ylab="Prob. success",xlab="",
        axes=FALSE,ylim=c(0.0,1))
axis(side=2)
axis(side=1,at=c(1:length(l.names)),l.names,las=2)
abline(h=0.6,lty="dashed",col="grey")
box()
legend("bottomleft",bty="n",paste("(",letters[i],") ",seas,sep=""))
dev.off()

q("no")

## Figure supplémentaire
dum=t(matrix(unlist(strsplit(names(countsim),"_")),nrow=2))
ndum=dum[,2]
modcol=ifelse(countsim>1,"red","blue")
pdf(file="count-sim_CMIP.pdf",width=12)
par(mar=c(10,4,1,1))
plot(countsim,type="h",axes=FALSE,ylab="Nb sim.",xlab="",col=modcol,lwd=3)
axis(side=2)
axis(side=1,at=c(1:length(countsim)),labels=ndum,las=2)
box()
dev.off()
