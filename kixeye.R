getwd()
setwd('kixeye/')
mydata = read.csv("data_output.csv")
pairs(mydata)
lm(mydata$retainted ~ mydata$tutorial)
glm.D = glm(mydata$retainted ~ mydata$tutorial + mydata$country + mydata$browser + mydata$date)
summary(glm.D)
l = lm(mydata$retainted ~ mydata$tutorial)
summary(l)
anova(l)
l = lm(mydata$retainted ~ mydata$country)
summary(l)
l = lm(mydata$retainted ~ mydata$tutorial)
l = glm(mydata$retainted ~ mydata$tutorial, family=binomial(logit))
summary(l)
plot(mydata$tutorial, mydata$retainted)
sum(mydata$tutorial==mydata$retainted)
plot(mydata$tutorial, mydata$retainted)
lc = glm(mydata$retainted ~ mydata$country, family=binomial(logit))
summary(lc)
drop1(lc, test='Chisq')
lc = glm(mydata$retainted ~ data.frame(mydata$country), family=binomial(logit))
data.frame(mydata$country)
mydata$country[1:100,]
mydata$country[1:100]
summary(lc)