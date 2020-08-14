library(ggplot2)
library(RColorBrewer)

myPalette <- brewer.pal(5, "Set2") 

data_train <- read.csv("train.csv", na.strings=c("", " ","NA"), stringsAsFactors = FALSE)

data_train$diagnosis <-  as.factor(data_train$diagnosis)

ggplot(data_train, aes(x = diagnosis)) +
  geom_bar(aes(y = stat(count)), color="black", fill=myPalette)

pie(table(data_train$diagnosis), labels = c("No DR","Mild","Moderate","Severe","Ploriferative DR"), border="white", col=myPalette)

table(data_train$diagnosis)
