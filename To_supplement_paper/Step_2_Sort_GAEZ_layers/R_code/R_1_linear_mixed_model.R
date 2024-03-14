library(lme4)
library(ggplot2)
library(dplyr)
library(merTools)
library(arm)
library(brms)

################# Preparing data and env #######################

# set cwd
cwd = paste0('C:/Users/Lenovo/OneDrive - Deakin University/',
            'PHD Progress/Paper_3/Earth_Engine_codes/',
            'Step_2_UNET_predict_future_urbanization/R_code')
setwd(cwd)


# get data
data = read.csv('../result_csv/GDP_Pop_NCP_urban_count.csv')
data_future = read.csv('../result_csv/SSP_GDP_Pop_future.csv')


############## build model and predict to the future ##########

# build model
model = lmer('count ~ GDP + Pop + (1+GDP|Province) + (1+Pop|Province)',data=data)
# model = lmer('count ~ scale(GDP) + scale(Pop) + (1|Province)',data=data)

# check the model by aligning pred with ture values
pred = predict(object = model,data)
true = data$count
df = data.frame(pred=pred,true=true,Province=data$Province)

# plot the comparison
df%>% 
  ggplot()+
  geom_point(aes(true,pred,color=Province))


############## predict to the future #############################

data_future$'count_pred' = predict(object = model,data_future)

ggplot() +
  geom_point(aes(x = year,y=count,color=Province),data = data) +
  geom_line(aes(x=year,y=mean,color=Province),data = 
                                                data_future%>% 
                                                group_by(Province,year) %>% 
                                                summarise(mean=mean(count_pred) ))



#_________________ build model using brms __________________##

model_brms = brm('count ~ GDP + Pop + (Pop + GDP|Province)',
                 data=data)


# prediction
pred_brms_original = predict(object = model_brms,data)
pred_brms_future = as.data.frame(predict(object = model_brms,data_future))

# making df for ploting
df_brms = data.frame(pred=pred_brms_original,true=true,Province=data$Province)
df_brms%>% 
  ggplot()+
  geom_point(aes(true,pred,color=Province))


data_future$'brms_future' = pred_brms_future$'Estimate'
data_future$'brms_future_low' = pred_brms_future$'Q2.5'
data_future$'brms_future_high' = pred_brms_future$'Q97.5'

ggplot() +
  geom_point(aes(x = year,y=count,color=Province),data = data) +
  geom_line(aes(x=year,y=mean_brms,color=Province),data = 
              data_future%>% 
              group_by(Province,year) %>% 
              summarise(mean_brms=mean(brms_future))) +
              
  geom_ribbon(aes(x=year,ymin=low,ymax=high,fill=Province),alpha=0.3,
              data = data_future%>% 
                     group_by(Province,year) %>% 
                     summarise(low=mean(brms_future_low),
                               high=mean(brms_future_high)))



# save prediction to disk
write.csv(x = data_future,
          file = '../result_csv/SSP_GDP_Pop_future_pred.csv',
          row.names = FALSE)




