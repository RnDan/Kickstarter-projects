## Objective (1): Predict whether a project would be successful or not based on attributes including 
#goal set for the project, category, country of launch, history of project owner's success etc.

# Using this analysis, Kicksarter can identify projects that would be successful based on various attributes 
# the project owner provides at the start of the project

# Source for project repository: https://webrobots.io/kickstarter-datasets/

## Data cleaning and transformation was performed in R
## Predictive Modelling was performed in Weka'
## Clustering was performed in R

# Read the consolidated kickstarter projects

Kickstarter_Projects = read.csv("/Kickstarter.csv")

# Basic datacleaning operation
# Removing redundant projects
unique_Kickstarter_Projects = unique(Kickstarter_Projects)

# Correcting format of date variable
unique_Kickstarter_Projects$deadline=as.Date(unique_Kickstarter_Projects$deadline, origin="1970-01-01")
unique_Kickstarter_Projects$state_changed_at=as.Date(unique_Kickstarter_Projects$state_changed_at, origin="1970-01-01")
unique_Kickstarter_Projects$created_at=as.Date(unique_Kickstarter_Projects$created_at, origin="1970-01-01")
unique_Kickstarter_Projects$launched_at=as.Date(unique_Kickstarter_Projects$launched_at, origin="1970-01-01")

#Transformation Step 1: Creation of additional attributes using date variables
unique_Kickstarter_Projects$days_since_launch = unique_Kickstarter_Projects$deadline - unique_Kickstarter_Projects$launched_at
unique_Kickstarter_Projects$days_creationToLaunch =unique_Kickstarter_Projects$launched_at-unique_Kickstarter_Projects$created_at
unique_Kickstarter_Projects$days_created_before_deadline =unique_Kickstarter_Projects$deadline-unique_Kickstarter_Projects$created_at

#install.packages("dplyr")
#library("dplyr")

#Shortlisting completed projects that would be used for analysis as live project does not provide a means for testing
Finished_Projects = filter(unique_Kickstarter_Projects,state!="live")
Finished_Projects$class = Finished_Projects$state

# CLASS VARIABLE DEFINITION:
    ## Successful - 1
    ## Failed/Canceled/Suspended - 0
Finished_Projects[Finished_Projects$class == "successful",]$class = 1
Finished_Projects[Finished_Projects$class == "failed",]$class = 0
Finished_Projects[Finished_Projects$class == "canceled",]$class = 0 
Finished_Projects[Finished_Projects$class == "suspended",]$class = 0  

# Program duration chosen as January 1 2015
Dataset_v1 = filter(Finished_Projects,launched_at>"2015-01-01")

# Transformation Step 2: project name column transformed based on Length of the project name as: Small, Medium, Long
Dataset_v1$length_name=sapply(gregexpr("\\W+", Dataset_v1$name), length) + 1
b <- c(-Inf, 6, 12, Inf)
names <- c("Small", "Medium", "Long")
Dataset_v1$length_name_class <- cut(Dataset_v1$length_name, breaks = b, labels = names)

# Transformation Step 3: Project Blurb column used to identify Sentiments and emotion
## they were added as new attributes for model

library(tm)
library(rvest)
library(SnowballC)
library(sentiment)
library(Rstem)

blurb=Corpus(VectorSource(Dataset_v1$blurb))
#create the toSpace content transformer
toSpace=content_transformer(function(x,pattern){return(gsub(pattern," ",x))})
blurb=tm_map(blurb,toSpace,"-")
blurb=tm_map(blurb,toSpace,":")
#remove punctuation
blurb=tm_map(blurb,removePunctuation)
#transform to lower case
blurb=tm_map(blurb,content_transformer(tolower))
#remove numbers
blurb=tm_map(blurb,removeNumbers)
#remove stopwords
blurb=tm_map(blurb,removeWords,stopwords("english"))
#remove additional whitespaces
blurb=tm_map(blurb,stripWhitespace)
#stem documents
blurb=tm_map(blurb,stemDocument)
cleaned_blurb=list()
count=0
for (i in 1:length(blurb)){
    count=count+i
    cleaned_blurb[i]=as.character(blurb[[i]])
}
blurb_2<-do.call("rbind", cleaned_blurb)
Dataset_v1 = cbind(Dataset_v1,blurb_2)
Dataset_v1$blurb_2=as.character(Dataset_v1$blurb_2)

# Sentiment analysis on cleaned blurb column

# Owing to large side of dataset parallel processing was done in R

library(parallel)
# Calculate the number of cores
no_cores <- detectCores() - 1
#clusters
cl<-makeCluster(no_cores)
sentiment<-function(x)
{
    library(sentiment)
    classify_polarity(as.character(x),algorithm = "bayes")[4]
}

Dataset_v1$name_pol =parLapply(cl,Dataset_v1[,29],sentiment)
Dataset_v1$name_pol = as.character(Dataset_v1$name_pol)

# Emotions based on blurb column

Emotion<-function(x)
{
    library(sentiment)
    classify_emotion(as.character(x),algorithm = "bayes")[7]
}

Dataset_v1$name_emo =parLapply(cl,Dataset_v1[,29],Emotion)
Dataset_v1$name_emo = as.character(Dataset_v1$name_emo)
Dataset_v1$name_emo[is.na(Dataset_v1$name_emo)] = "unknown"

# Using the above transformations, decision tree model had an accuracy of ~70%
# Below step was performed to improve accuracy

# Transformation step 4:Identify project owner history in term of number of successful vs unsuccessful project

library("sqldf")
library("stringr")


Dataset_v2=sqldf('select a.*,b.number_projects_creator, c.number_projects_successful 
                            from Dataset_v1 a left join 
                            (select creator,count(*) as number_projects_creator 
                                from Dataset_v1 group by creator)b 
                                on a.creator=b.creator left join 
                                    (select creator,count(*) as number_projects_successful 
                                        from Dataset_v1 where state="successful" group by creator)c on a.creator=c.creator')


Dataset_v2$Prior_project_by_creator = Dataset_v2$number_projects_creator-1
Dataset_v2$Prior_successful_proj_by_creator=Dataset_v2$number_projects_successful-1

# Replacing NA values from calculation with 0
Dataset_v2$Prior_project_by_creator[is.na(Dataset_v2$Prior_project_by_creator)] = 0
Dataset_v2$Prior_successful_proj_by_creator[is.na(Dataset_v2$Prior_successful_proj_by_creator)] = 0
Dataset_v2$History_of_successful_projects = ifelse(Dataset_v2$Prior_successful_proj_by_creator >0, "Yes", "No")
Dataset_v2$goal[is.na(Dataset_v2$goal)] = 0
Dataset_v2$Prior_project_by_creator[is.na(Dataset_v2$Prior_project_by_creator)] = 0

Final_dataset = Dataset_v2

# The final dataset was used for Classification models using Weka

# Data was split as 70% for training and 30% for test datasets

# Weka Results
## Decision Tree Model on Test Data
        # Accuracy - 74%
        # Precision - 82%
        # Recall - 58%
        #F1 measure - 68%

## K Nearest Neighbour (K=11) on Test Data
        # Accuracy - 70%
        # Precision - 70%
        # Recall - 69.3%
        #F1 measure - 70%

## Naive Bayes on Test Data (Dataset had correlated attributes)
        # Accuracy - 63%
        # Precision - 59%
        # Recall - 80%
        #F1 measure - 68%

## Multilayer Perceptron on Test Data
        # Accuracy - 73%
        # Precision - 75%
        # Recall - 68%
        #F1 measure - 71%