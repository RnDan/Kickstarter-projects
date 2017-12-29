# Since the dataset contains both continuous and categorical variable, gower distance is used as distance measurement

Final_dataset_dist=daisy(Final_dataset[1:10000,],metric="gower")
Final_dataset_mat=as.matrix(Final_dataset_dist)

#To identify clustering tendency, hopkins statistics were used

#H = 0.4, so there is only slight clustering tendency

hopkin_fn = function(x)
{
    library(clustertend)
    hopkins(x,n=nrow(x)-1)
}
h=hopkins(Final_dataset_mat,n=nrow(Final_dataset_mat)-1)

## K-Medoid clustering - Since the dataset contains both continuous and categorical values, K-Medoid clustering was used using PAM function
# Checking best K using silhouette coeff plot

sil_width <- vector('numeric')
for(i in 2:20){
    
    pam_fit <- pam(Final_dataset_dist,
                   diss = TRUE,
                   k = i)
    
    sil_width[i-1] <- pam_fit$silinfo$avg.width
    
}

k=1:19
plot(k,sil_width,type="b")

#Best k was obtained at k=11
#fviz_cluster was used to plot the clusters

library("factoextra")
my.cluster_pam <- pam(Final_dataset_sample, k=11)
fviz_cluster(my.cluster_pam, main = "K-Medoid", ellipse = FALSE,
             geom = "point", palette = "tol", ggtheme = theme_classic())

# Clustering using DBScan technique
# DBScan is a density based clustering technique
# For this analysis, DBScan gave more separable clusters when compared to K-Medoid clustering

library(dbscan)
my_cluster_dbscan =dbscan(final_v5_dist2,0.147, minPts=40)
fviz_cluster(my_cluster_dbscan, final_v5_dist2, main = "DBSCAN", frame = FALSE,labelsize=0,
             ellipse = TRUE, show.clust.cent = TRUE,
             geom = "point",palette = "tol", ggtheme = theme_classic())