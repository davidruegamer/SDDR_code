library("Metrics")
library("rmatio")

d=read.mat("resVAFC2.mat")
d=d$retts2
colo=read.mat("../colon_train_test.mat")
leuke=read.mat("../leukemia_train_test.mat")
duke=read.mat("../duke_train_test.mat")

colo$Yts[colo$Yts== -1] = 0
leuke$Yts[colo$Yts== -1] = 0
duke$Yts[colo$Yts== -1] = 0
duke$Yts=duke$Yts[1:4]

aucscolo=matrix(NA,20,2)

for(ii in 1:20)
  {
  for(jj in 1:2)
    {
    aucscolo[ii,jj]=auc(colo$Yts,d$colon[[jj]][,ii])
    }
  }


aucsleuke=matrix(NA,20,2)

for(ii in 1:20)
{
  for(jj in 1:2)
  {
    aucsleuke[ii,jj]=auc(leuke$Yts,d$leukemia[[jj]][,ii])
  }
}

aucsduke=matrix(NA,20,2)

for(ii in 1:20)
{
  for(jj in 1:2)
  {
    aucsduke[ii,jj]=auc(duke$Yts,d$duke[[jj]][,ii])
  }
}

colMeans(aucscolo)
apply(aucscolo,FUN="sd",MAR=2)

colMeans(aucsleuke)
apply(aucsleuke,FUN="sd",MAR=2)

colMeans(aucsduke)
apply(aucsduke,FUN="sd",MAR=2)




