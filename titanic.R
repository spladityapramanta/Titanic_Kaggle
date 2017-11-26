train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Feature Engineering ############################################
# Create Survived Feature in Test Data Frame
test$Survived <- NA

# Combine test and train to create the engineered features
combi <- rbind(train,test)

#################################################################
# 1. GROUP BY TITLE
# We want to group the name by its title
# first convert from factor to character type
combi$Name <- as.character(combi$Name)

# split the name to get the title and put it in new column
# Format: Family name, Title. Surname
combi$Title <- sapply(combi$Name, FUN = function(x){strsplit(x,split = '[,.]')[[1]][2]})

# strip of spaces
combi$Title <- sub(' ', '',combi$Title)

# check its content
table(combi$Title)

# aggregate similar title
combi$Title[combi$Title %in% c('Mlle','Mme')] <- 'Mlle'
combi$Title[combi$Title %in% c('Capt','Don','Major','Sir')] <- 'Sir'
combi$Title[combi$Title %in% c('Dona','Lady','the Countess','Jonkheer')] <- 'Lady'

# take it back as factor type
combi$Title <- factor(combi$Title)
###############################################################
# 2. Family Size

combi$FamilySize <- combi$SibSp + combi $Parch + 1 # 1 himself

# group the family with their surname (as it might be several similar family name)
combi$Surname <- sapply(combi$Name, FUN = function(x){strsplit(x, split = '[,.]')[[1]][1]})
combi$FamilyId <- paste(as.character(combi$FamilySize), combi$Surname, sep='')

# group family with size 2 in one group
combi$FamilyId[combi$FamilySize <2 ] <-'Small'

table(combi$FamilyId)

# there are some data that should be in small group
famIDs <- data.frame(table(combi$FamilyId))
famIDs <- famIDs[famIDs$Freq <= 2,]
combi$FamilyId[combi$FamilyId %in% famIDs$Var1] <- 'Small'
combi$FamilyId <- factor(combi$FamilyId)
###########################################################


###########################################################
# MACHINE LEARNING PART
# Handling NA using decision tree
library(rpart)
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                data=combi[!is.na(combi$Age),], 
                method="anova")
combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])

# Handling NA using median insertion
combi$Fare[which(is.na(combi$Fare))] <- median(combi$Fare, na.rm=TRUE)

# Handding missing value
combi$Embarked[which(combi$Embarked == '')] = "S"
combi$Embarked <- factor(combi$Embarked)

# Spliting data
train <- combi[1:891,]
test <- combi[892:1309,]

# Using conditional inference random forest
library(party)
set.seed(415)
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                 Embarked + Title + FamilySize + FamilyId,
               data = train, 
               controls=cforest_unbiased(ntree=2000, mtry=3))

Prediction <- predict(fit, test, OOB=TRUE, type = "response")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "forest_cond_inf.csv", row.names = FALSE)
##########################################################