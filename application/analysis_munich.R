library(imager)
library(readr)
library(dplyr)
library(tidyr)
library(abind)
library(ggplot2)
library(ggmap)
library(viridis)
library(grid)
library(xtable)
library(deepregression)

nr_words = 1000
embedding_size = 100
maxlen = 50

cnn_block <- function(filters, kernel_size, pool_size, rate, input_shape = NULL){
  function(x){
    x %>%
      layer_conv_2d(filters, kernel_size, padding="same", input_shape = input_shape) %>%
      layer_activation(activation = "relu") %>%
      layer_batch_normalization() %>%
      layer_max_pooling_2d(pool_size = pool_size) %>%
      layer_dropout(rate = rate)
  }
}

if(!file.exists("airbnb/munich_clean.RDS"))
{
  
  # reticulate::source_python("airbnb/resize_img.py")
  # resize_img("airbnb/data/pictures/32/", tuple(200,200))
  
  list_images <- lapply(list.files("airbnb/data/pictures/32/", full.names = T), 
                        function(x) tryCatch(load.image(x), error = function(e) x))
  no_image <- sapply(list_images, is.character)
  no_image_id <- sapply(list_images[no_image], 
                        function(x)gsub(".*pictures/32//(.*)\\.jpg","\\1", x))
  
  list_images <- list_images[!no_image]
  
  d <- read_csv("airbnb/data/germany_bv_munichlistings.csv")
  d <- d %>% filter(!id%in%no_image_id)
  
  # check same lengths
  if(!nrow(d)==length(list_images)) stop("Check images / tab data")
  
  d <- d %>% filter(!is.na(review_scores_rating))
  
  d$room_type <- as.factor(d$room_type)
  d$bedrooms[is.na(d$bedrooms)] <- 0
  d$bedrooms <- as.factor(d$bedrooms)
  d$beds[is.na(d$beds)] <- 0
  d$beds <- as.factor(d$beds)
  
  saveRDS(d, "airbnb/munich_clean.RDS")
  
}else if(!file.exists("munich_clean_text.RDS")){
  
  d <- readRDS("airbnb/munich_clean.RDS")
  
}
# # descriptives
# hist(d$price, breaks = 100)
# 
# # an intercept + geo model
# mod_int <- deepregression(y = d$price, family = "pareto_ls",
#                           list_of_formulae = list(~1 + s(latitude, longitude), 
#                                                   ~1), 
#                           data = d,
#                           list_of_deep_models = NULL,
#                           optimizer = optimizer_rmsprop())
# 
# mod_int %>% fit(epochs = 500, 
#                 early_stopping = TRUE, 
#                 patience = 40,
#                 validation_split = 0.1)
# 
# central_station_loc <- c(48.1402669, 11.559998)
# # => have a nicer geo map and show central station
# 
# # a more complex model
# mod_int <- deepregression(y = d$price, 
#                           family = "pareto_ls",
#                           list_of_formulae = list(
#                             ~1 + s(latitude, longitude) + 
#                               room_type + s(accommodates) + 
#                               bedrooms + beds + 
#                               s(review_scores_rating), 
#                             ~1), 
#                           data = d,
#                           list_of_deep_models = NULL,
#                           optimizer = optimizer_rmsprop(), 
#                           tf_seed = 3,
#                           df = c(30, 8, 8))
# 
# mod_int %>% fit(epochs = 500, 
#                 early_stopping = TRUE, 
#                 patience = 40,
#                 validation_split = 0.1)
# 
# mod_int %>% plot()
# mod_int %>% coef(params = 1, type = "linear")

  
  
if(!file.exists("munich_clean_text.RDS")){
  
  # description 
  
  library(tidytext)
  library(tm)
  
  tokenizer <- text_tokenizer(num_words = nr_words)
  
  # remove stopwords
  data("stop_words")
  stopwords_regex = paste(c(stopwords('en'), stop_words$word), 
                          collapse = '\\b|\\b')
  stopwords_regex = paste0('\\b', stopwords_regex, '\\b')
  d$description = tolower(d$description)
  d$description = stringr::str_replace_all(d$description, stopwords_regex, '')
  d$description = gsub('[[:punct:] ]+',' ', d$description)
  
  saveRDS(d$description, file="munich_description.RDS")
  
  tokenizer %>% fit_text_tokenizer(d$description)
  
  # text to sequence
  text_seqs <- texts_to_sequences(tokenizer, d$description)
  
  # pad text sequences
  text_padded <- text_seqs %>%
    pad_sequences(maxlen = maxlen)
  
  # save words for later
  words <- data_frame(
    word = names(tokenizer$word_index), 
    id = as.integer(unlist(tokenizer$word_index))
  )
  
  words <- words %>%
    filter(id <= tokenizer$num_words) %>%
    arrange(id)
  
  saveRDS(words, file="munich_words.RDS")
  rm(words)
  gc()
  
  # text sequences as list of one array
  text_embd <- list(texts = array(text_padded, dim=c(NROW(d), maxlen)) )
  
  # create input list
  d <- append(d, text_embd) 
  
  saveRDS(d, file="munich_clean_text.RDS")
  
  rm(text_embd)
  gc()
  
  
}else{
  
  d <- readRDS("munich_clean_text.RDS")
  
}

desc <- d$texts
d$texts <- NULL
d <- as.data.frame(d)

nn_big <- function(x){
  application_resnet50(
    input_tensor = x,
    weights='imagenet',
    include_top = FALSE)$output %>%
    layer_global_average_pooling_2d()
}

# nn_big <- function(x) x %>%
#   # Conv-Block 1
#   layer_conv_2d(filters = 4, kernel_size = c(3,3), activation= "relu",
#                 input_shape = shape(200, 200, 3),
#                 kernel_regularizer=regularizer_l2(l = 0.0001)) %>%
#   layer_batch_normalization() %>%
#   layer_max_pooling_2d(pool_size = c(2,2)) %>%
#   layer_dropout(rate = 0.4) %>%
#   # FC layer
#   layer_flatten() %>%
#   layer_dense(units = 16, activation = "relu")

cnn1 <- cnn_block(filters = 32, kernel_size = c(3,3), pool_size = c(3,3), rate = 0.25)
cnn2 <- cnn_block(filters = 64, kernel_size = c(3,3), pool_size = c(2,2), rate = 0.25)
cnn3 <- cnn_block(filters = 128, kernel_size = c(3,3), pool_size = c(2,2), rate = 0.25)
cnn4 <- cnn_block(filters = 64, kernel_size = c(3,3), pool_size = c(2,2), rate = 0.25)
  
nn_big <- function(x) x %>% 
  cnn1() %>%
  cnn2() %>%
  cnn3() %>%
  layer_flatten() %>%
  layer_dense(128) %>%
  layer_activation(activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(64) 

nn_big_top <- function(x) x %>% layer_dense(units=1
                                            , kernel_initializer = 'zeros'
                                            )

nn_fc <- function(x) x %>% 
  layer_dense(units = 16, activation = "relu", name = "fc1") %>% 
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 4, activation = "relu", name = "fc2") %>% 
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 2, activation = "relu", name = "fc3")

nn_fc_var <- nn_fc

d$image <- paste0("/home/david/airbnb/airbnb/data/pictures/32/",
                   d$id, ".jpg")

rt <- model.matrix(~ room_type, data=d)[,-1]
colnames(rt) <- gsub(" ", "_", colnames(rt))
d <- cbind(d, rt)
d <- cbind(d, model.matrix(~ beds, data=d)[,-1])
d <- cbind(d, model.matrix(~ bedrooms, data=d)[,-1])

d$days_since_last_review <- as.numeric(difftime(d$date, d$last_review))
desc <- desc[!is.na(d$review_scores_location),]
d <- d %>% filter(!is.na(review_scores_location))

d$beds <- factor(pmin(as.numeric(as.character(d$beds)), 6), labels = 0:6)
d$bedrooms <- factor(pmin(as.numeric(as.character(d$bedrooms)), 4), labels = 0:4)

set.seed(41)

ind_train <- sample(1:nrow(d), round(0.9*nrow(d)))
ind_test <- setdiff(1:nrow(d), ind_train)

d_train <- d %>% slice(ind_train)
d_test <- d %>% slice(ind_test)
d_train$desc <- desc[ind_train,]
d_test$desc <- desc[ind_test,]

tokenizer <- text_tokenizer(num_words = nr_words)
tokenizer %>% fit_text_tokenizer(readRDS("munich_description.RDS"))

# modeling
embd_mod <- function(x) x %>%
  layer_embedding(input_dim = tokenizer$num_words,
                  output_dim = embedding_size) %>%
  # layer_lambda(f = function(x) k_mean(x, axis = 2)) %>%
  # layer_dense(10) %>% 
  layer_flatten() 


############################################################

mod_struct <- deepregression(y = d_train$price, 
                             family = "log_normal",
                             list_of_formulae = list(
                               ~1 + te(latitude, longitude, k = c(8,8)) + 
                                 room_type + s(accommodates) + 
                                 bedrooms + beds + 
                                 s(days_since_last_review) + 
                                 s(reviews_per_month, k = 25) + 
                                 s(review_scores_rating), 
                               ~1 + room_type + te(latitude, longitude, k = c(8,8))), 
                             data = d_train,
                             list_of_deep_models = NULL,
                             optimizer = optimizer_adam(), 
                             tf_seed = 6,
                             split_fun = function(x) list(x, nn_big_top),
                             df = list(list(c(6,6), 8, 6, 6, 8), 
                                       list(c(6,6))))


mod_struct %>% fit(epochs = 4000, 
                   view_metrics = FALSE,
                   early_stopping = TRUE, 
                   patience = 25,
                   batch_size = 32,
                   validation_split = 0.1)

fitted_vals_str <- mod_struct %>% fitted()
cor(log(fitted_vals_str), log(d_train$price))
par(mfrow=c(1,1))
plot(log(fitted_vals_str), log(d_train$price))
abline(0,1,col="red")
pred_vals_str <- mod_struct %>% predict(d_test)

rbind(data.frame(truth = log(d_train$price), 
                 prediction = log(fitted_vals_str[,1]))) %>% gather() %>% 
  ggplot(aes(x = value, fill = key)) + geom_density(alpha=0.2) + theme_bw()

res_df <- data.frame(model = "structured", 
                           cor_train = cor(log(d_train$price), log(fitted_vals_str[,1])),
                           cor_test = cor(log(d_test$price), log(pred_vals_str[,1])))


############################################################

mod_struct_dnn <- deepregression(y = d_train$price, 
                                 family = "log_normal",
                                 list_of_formulae = list(
                                   ~1 + te(latitude, longitude, k = c(8,8)) + 
                                     room_type + s(accommodates) + 
                                     bedrooms + beds + 
                                     s(days_since_last_review) + 
                                     s(reviews_per_month, k = 25) + 
                                     s(review_scores_rating) + 
                                     nn_fc(beds1, beds2, beds3, beds4, beds5, beds6, 
                                           bedrooms1, bedrooms2, bedrooms3, bedrooms4, 
                                           room_typeHotel_room, room_typePrivate_room, 
                                           room_typeShared_room) %OZ% (room_type + bedrooms + beds), 
                                   ~1 + room_type + te(latitude, longitude, k = c(8,8))), 
                                 data = d_train,
                                 list_of_deep_models = list(nn_fc = nn_fc),
                                 optimizer = optimizer_adam(), 
                                 tf_seed = 6,
                                 split_fun = function(x) list(x, nn_big_top),
                                 df = list(list(c(6,6), 8, 6, 6, 8), 
                                           list(c(6,6))))


mod_struct_dnn %>% fit(epochs = 4000, 
                       view_metrics = FALSE,
                       early_stopping = TRUE, 
                       patience = 25,
                       batch_size = 32,
                       validation_split = 0.1)

fitted_vals_str_nn <- mod_struct_dnn %>% fitted()
cor(log(fitted_vals_str_nn), log(d_train$price))
par(mfrow=c(1,1))
plot(log(fitted_vals_str_nn), log(d_train$price))
abline(0,1,col="red")
pred_vals_str_nn <- mod_struct_dnn %>% predict(d_test)

res_df <- rbind(res_df,
                data.frame(model = "structured w/ dnn", 
                           cor_train = cor(log(d_train$price), log(fitted_vals_str_nn[,1])),
                           cor_test = cor(log(d_test$price), log(pred_vals_str_nn[,1])))
)

############################################################
# w/o OZ

mod_int <- deepregression(y = d_train$price, 
                          family = "log_normal",
                          list_of_formulae = list(
                            ~1 + te(latitude, longitude, k = c(8,8)) + 
                              room_type + s(accommodates) + 
                              bedrooms + beds + 
                              s(days_since_last_review) + 
                              s(reviews_per_month, k = 25) + 
                              s(review_scores_rating) + 
                              nn(image)+ 
                              embd_mod(desc) +
                              nn_fc(beds1, beds2, beds3, beds4, beds5, beds6,
                                    bedrooms1, bedrooms2, bedrooms3, bedrooms4,
                                    room_typeHotel_room, room_typePrivate_room,
                                    room_typeShared_room) %OZ% (room_type + bedrooms + beds)
                            , 
                            ~1 + room_type + te(latitude, longitude, k = c(8,8))), 
                          data = d_train,
                          image_var = list(image = list(c(200,200,3))),
                          list_of_deep_models = list(nn = nn_big, 
                                                     embd_mod = embd_mod,
                                                     nn_fc = nn_fc
                          ),
                          optimizer = optimizer_adam(lr = 0.0001), 
                          tf_seed = 6,
                          split_fun = function(x) list(x, nn_big_top),
                          df = list(list(c(6,6), 8, 6, 6, 8), 
                                    list(c(6,6))))

if(file.exists("weights_airbnb_model.hdf5")){
  mod_int$model$load_weights(filepath="weights_airbnb_model.hdf5", by_name = FALSE)
  
}else{
  
  mod_int$model$get_layer(name="structured_nonlinear_1")$set_weights(
    mod_struct$model$get_layer(name="structured_nonlinear_1")$get_weights()
  )
  
  mod_int$model$get_layer(name="structured_nonlinear_2")$set_weights(
    mod_struct$model$get_layer(name="structured_nonlinear_2")$get_weights()
  )
  
  mod_int$model$get_layer(name="fc1")$set_weights(
    mod_struct_dnn$model$get_layer(name="fc1")$get_weights()
  )
  
  mod_int$model$get_layer(name="fc2")$set_weights(
    mod_struct_dnn$model$get_layer(name="fc2")$get_weights()
  )
  
  mod_int$model$get_layer(name="fc3")$set_weights(
    mod_struct_dnn$model$get_layer(name="fc3")$get_weights()
  )
  
  
  
  mod_int %>% fit(epochs = 5000, 
                  view_metrics = FALSE,
                  early_stopping = TRUE, 
                  patience = 2,
                  batch_size = 32,
                  validation_split = 0.1)
  
  save_model_weights_hdf5(mod_int$model, filepath="weights_airbnb_model.hdf5")
  # .rs.restartR()
  
}

fitted_vals <- mod_int %>% fitted() %>% unlist()

pred_test <- mod_int %>% predict(d_test) %>% unlist()

res_df <- rbind(res_df,
                data.frame(model = "main_wooz", 
                           cor_train = cor(log(d_train$price), log(fitted_vals)),
                           cor_test = cor(log(d_test$price), log(pred_test)))
)


# w/ OZ


mod_int_woz <- deepregression(y = d_train$price, 
                          family = "log_normal",
                          list_of_formulae = list(
                            ~1 + te(latitude, longitude, k = c(8,8)) + 
                              room_type + s(accommodates) + 
                              bedrooms + beds + 
                              s(days_since_last_review) + 
                              s(reviews_per_month, k = 25) + 
                              s(review_scores_rating) + 
                              nn(image) %OZ% (room_type + bedrooms + beds + 
                                                te(latitude, longitude, k = c(8,8)) +
                                                s(accommodates) +
                                                s(days_since_last_review) +
                                                s(reviews_per_month, k = 25) +
                                                s(review_scores_rating)) + 
                              embd_mod(desc) +
                              nn_fc(beds1, beds2, beds3, beds4, beds5, beds6,
                                    bedrooms1, bedrooms2, bedrooms3, bedrooms4,
                                    room_typeHotel_room, room_typePrivate_room,
                                    room_typeShared_room) %OZ% (room_type + bedrooms + beds)
                            , 
                            ~1 + room_type + te(latitude, longitude, k = c(8,8))), 
                          data = d_train,
                          image_var = list(image = list(c(200,200,3))),
                          list_of_deep_models = list(nn = nn_big, 
                                                     embd_mod = embd_mod,
                                                     nn_fc = nn_fc
                          ),
                          optimizer = optimizer_adam(lr = 0.0001), 
                          tf_seed = 6,
                          split_fun = function(x) list(x, nn_big_top),
                          df = list(list(c(6,6), 8, 6, 6, 8), 
                                    list(c(6,6))))

if(file.exists("weights_airbnb_model_woz.hdf5")){
  
  mod_int_woz$model$load_weights(filepath="weights_airbnb_model_woz.hdf5", by_name = FALSE)
  
}else{
  
  mod_int_woz$model$get_layer(name="structured_nonlinear_1")$set_weights(
    mod_struct$model$get_layer(name="structured_nonlinear_1")$get_weights()
  )
  
  mod_int_woz$model$get_layer(name="structured_nonlinear_2")$set_weights(
    mod_struct$model$get_layer(name="structured_nonlinear_2")$get_weights()
  )
  
  mod_int_woz$model$get_layer(name="fc1")$set_weights(
    mod_struct_dnn$model$get_layer(name="fc1")$get_weights()
  )
  
  mod_int_woz$model$get_layer(name="fc2")$set_weights(
    mod_struct_dnn$model$get_layer(name="fc2")$get_weights()
  )
  
  mod_int_woz$model$get_layer(name="fc3")$set_weights(
    mod_struct_dnn$model$get_layer(name="fc3")$get_weights()
  )
  
  
  
  mod_int_woz %>% fit(epochs = 5000, 
                  view_metrics = FALSE,
                  early_stopping = TRUE, 
                  patience = 2,
                  batch_size = 32,
                  validation_split = 0.1)
  
  save_model_weights_hdf5(mod_int_woz$model, filepath="weights_airbnb_model_woz.hdf5")
  # .rs.restartR()
  
}


fitted_vals <- mod_int_woz %>% fitted() %>% unlist()

pred_test <- mod_int_woz %>% predict(d_test) %>% unlist()

res_df <- rbind(res_df,
                data.frame(model = "main_woz", 
                           cor_train = cor(log(d_train$price), log(fitted_vals)),
                           cor_test = cor(log(d_test$price), log(pred_test)))
)



par(mfrow=c(2,2))
peplots <- plot(mod_int_woz, type="b", which = 2:5, plot = F)
pepdata <- do.call("rbind", lapply(peplots[2:5], 
                                   function(x) data.frame(value = x$value[,1],
                                                          pe = x$partial_effect[,1],
                                                          name = x$org_feature_name)))
levels(pepdata$name) <- c("Accommodates", "Days since last review", 
                          "Reviews per month", "Review rating")
ggplot(pepdata, aes(x = value, y = exp(pe))) + 
  geom_line(size=1) +
  geom_point(size=6, colour="white") + 
  geom_point(size=3) + 
  facet_wrap(~name, scales = "free") + 
  theme_bw() + xlab("Feature value") + ylab("Partial multiplicative effect") + 
  theme(text = element_text(size=16)) + 
  ggsave(file = "peplots.pdf")

cbind(mean_effect = coef(mod_int_woz, type = "linear")$location,
      scale_effect = c(exp(coef(mod_int_woz, type = "linear")$scale), rep(0,10))) %>% 
  xtable()
plot(mod_int_woz, which_param = 2)

par(mfrow=c(1,1))
plot(log(fitted_vals),log(d_train$price))
abline(0,1,col="red")



pe_te <- get_partial_effect(mod_int_woz, name = "latitude", newdata = d)
pe_te2 <- get_partial_effect(mod_int_woz, name = "latitude", newdata = d, which_param=2)
prices_loc <- cbind(price = pe_te[,1], variance = pe_te2[,1], d %>% select(latitude, longitude))

#####

myLocation<-c(min(d$longitude), min(d$latitude), max(d$longitude), max(d$latitude))
myMap <- get_map(location = myLocation, 
                 source="google", 
                 maptype="roadmap", 
                 crop=TRUE)
ggmap(myMap) + # xlab("Longitude") + ylab("Latitude") + 
  geom_point(data = prices_loc, aes(x = longitude, y = latitude, 
                                    colour = exp(price), alpha=0.005)) + 
  scale_colour_viridis_c(option = 'magma', direction = -1, 
                         name = "Multiplicative Effect on the\nPrice Distrbution's Mean") + 
  guides(alpha = FALSE) + ggtitle("Geographic Location Effect") +
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size=10),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        legend.position = "bottom",
        legend.key.width=unit(1.2,"cm")) + 
  ggsave(file = "munich.pdf")

ggmap(myMap) + # xlab("Longitude") + ylab("Latitude") + 
  geom_point(data = prices_loc, aes(x = longitude, y = latitude, 
                                    colour = exp(variance), alpha=0.005)) + 
  scale_colour_viridis_c(option = 'magma', direction = -1, 
                         name = "Multiplicative Effect on the\nPrice Distribution's Scale") + 
  guides(alpha = FALSE) + ggtitle("Geographic Location Effect") +
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size=10),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        legend.position = "bottom",
        legend.key.width=unit(1.2,"cm")) + 
  ggsave(file = "munich_scale.pdf")

### image effect

d_dummy <- d[,c("latitude", "longitude", "room_type", "accommodates", 
                "bedrooms", "beds", "days_since_last_review", 
                "reviews_per_month", "review_scores_rating", 
                "image")]

d_dummy$latitude <- mean(d$latitude)
d_dummy$longitude <- mean(d$longitude)
d_dummy$days_since_last_review <- mean(d$days_since_last_review)
d_dummy$reviews_per_month <- mean(d$reviews_per_month)
d_dummy$review_scores_rating <- mean(d$review_scores_rating)
d_dummy$room_type <- factor(names(which.max(table(d$room_type))), levels = levels(d$room_type))
d_dummy$beds <- factor(names(which.max(table(d$beds))), levels = levels(d$beds))
d_dummy$bedrooms <- factor(names(which.max(table(d$bedrooms))), levels = levels(d$bedrooms))
d_dummy$desc <- ""
d_dummy[, c("beds1", "beds2", "beds3", "beds4", "beds5", "beds6", 
            "bedrooms1", "bedrooms2", "bedrooms3", "bedrooms4", 
            "room_typeHotel_room", "room_typePrivate_room", 
            "room_typeShared_room")] <- data.frame(c(1,0,0,0,0,0,
                                                     1,0,0,0,
                                                     0,0,0))[rep(1,nrow(d)),]

text_seqs <- texts_to_sequences(tokenizer, d_dummy$desc)
# pad text sequences
text_padded <- text_seqs %>%
  pad_sequences(maxlen = maxlen)
d_dummy$desc <- text_padded

pred_images <- mod_int %>% predict(newdata = d_dummy) %>% unlist()
saveRDS(pred_images, file="pred_images.RDS")

qs <- stats::quantile(pred_images, seq(0, 1, l = 5))

qpics <- c()
for(i in 1:length(qs)){ qpics <- c(qpics, 
                                   d$image[which(abs(pred_images - qs[i]) == 
                                                   (min(abs(pred_images - qs[i]), na.rm = TRUE)))[1]]
                                   )
}
  
piclist <- list()
i=1
par(mfrow=c(2,4))
for(pic in qpics){
  plot(piclist[[i]] <- load.image(pic), axes=FALSE)
  i <- i+1
}

vals_pics <- qs

summary(pred_images)
piclist <- lapply(piclist, function(x)rasterGrob(x, interpolate=TRUE))

ymin <- 0
ymax <- 0.004
size_half <- 10

ggplot(data.frame(predval = (pred_images)), aes(x=predval)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  xlab("Partial effect of the DNN") + ylab("Frequency") + 
  geom_density(alpha=.2, fill="#FF6666") +
  # ylim(0,0.006) +
  annotation_custom(piclist[[1]], 
                    xmin=(vals_pics[1])-size_half, 
                    xmax=(vals_pics[1])+size_half+10, 
                    ymin=ymin, ymax=ymax) + 
  annotation_custom(piclist[[2]], 
                    xmin=(vals_pics[2])-size_half-15, 
                    xmax=(vals_pics[2])+size_half-5, 
                    ymin=ymin, ymax=ymax) + 
  annotation_custom(piclist[[3]], 
                    xmin=(vals_pics[3])-size_half+10, 
                    xmax=(vals_pics[3])+size_half+20, 
                    ymin=ymin, ymax=ymax) + 
  annotation_custom(piclist[[4]], 
                    xmin=(vals_pics[4])-size_half+25, 
                    xmax=(vals_pics[4])+size_half+35, 
                    ymin=ymin, ymax=ymax) + 
  annotation_custom(piclist[[5]], 
                    xmin=(vals_pics[5])-size_half-10, 
                    xmax=(vals_pics[5])+size_half, 
                    ymin=ymin, ymax=ymax) + theme_bw() + 
  ggsave(file = "pic_effect.pdf")
  

cor(pred_images, d$price)
############################################################

d_dummy$desc <- d$description
text_seqs <- texts_to_sequences(tokenizer, d_dummy$desc)
# pad text sequences
text_padded <- text_seqs %>%
  pad_sequences(maxlen = maxlen)
d_dummy$desc <- text_padded
d_dummy$image <- qpics[4]
pred_texts <- mod_int %>% predict(newdata = d_dummy) %>% unlist()
saveRDS(pred_texts, file="pred_texts.RDS")

cor(pred_texts, d$price)

chosen_text <- c(1,round(nrow(d)*c(0.1,0.25,0.5,0.75,0.9,0.95)),nrow(d))
qtexts <- d$desc[which(order(pred_texts) %in% chosen_text)]
as.character(qtexts)



###########################################################################################

mod_struct_dnn_text <- deepregression(y = d_train$price, 
                                      family = "log_normal",
                                      list_of_formulae = list(
                                        ~1 + te(latitude, longitude, k = c(8,8)) + 
                                          room_type + s(accommodates) + 
                                          bedrooms + beds + 
                                          s(days_since_last_review) + 
                                          s(reviews_per_month, k = 25) + 
                                          s(review_scores_rating) + 
                                          embd_mod(desc) %OZ% (room_type + bedrooms + beds + 
                                                                 te(latitude, longitude, k = c(8,8)) +
                                                                 s(accommodates) +
                                                                 s(days_since_last_review) +
                                                                 s(reviews_per_month, k = 25) +
                                                                 s(review_scores_rating)) + 
                                          nn_fc(beds1, beds2, beds3, beds4, beds5, beds6, 
                                                bedrooms1, bedrooms2, bedrooms3, bedrooms4, 
                                                room_typeHotel_room, room_typePrivate_room, 
                                                room_typeShared_room) %OZ% 
                                          (room_type + bedrooms + beds), 
                                        ~1 + room_type + te(latitude, longitude, k = c(8,8))), 
                                      data = d_train,
                                      list_of_deep_models = list(nn_fc = nn_fc,
                                                                 embd_mod = embd_mod),
                                      optimizer = optimizer_adam(), 
                                      tf_seed = 6,
                                      split_fun = function(x) list(x, nn_big_top),
                                      df = list(list(c(6,6), 8, 6, 6, 8), 
                                                list(c(6,6))))


mod_struct_dnn_text %>% fit(epochs = 4000, 
                            view_metrics = FALSE,
                            early_stopping = TRUE, 
                            patience = 25,
                            batch_size = 32,
                            validation_split = 0.1)

fitted_vals_str_nnt <- mod_struct_dnn_text %>% fitted()
cor(log(fitted_vals_str_nnt), log(d_train$price))
par(mfrow=c(1,1))
plot(log(fitted_vals_str_nnt), log(d_train$price))
abline(0,1,col="red")
pred_vals_str_nnt <- mod_struct_dnn_text %>% predict(d_test)

res_df <- rbind(res_df,
                data.frame(model = "structured w/ dnn + embd", 
                           cor_train = cor(log(d_train$price), log(fitted_vals_str_nnt[,1])),
                           cor_test = cor(log(d_test$price), log(pred_vals_str_nnt[,1])))
)

#################################################################################


mod_pic <- deepregression(y = d_train$price, 
                          family = "log_normal",
                          list_of_formulae = list(
                            ~1 + nn(image), 
                            ~1), 
                          data = d_train,
                          image_var = list(image = list(c(200,200,3))),
                          list_of_deep_models = list(nn = nn_big#, 
                                                     #embd_mod = embd_mod,
                                                     # nn_fc = nn_fc#,
                                                     #nn_fc_var = nn_fc_var
                          ),
                          optimizer = optimizer_adam(lr = 0.00001), 
                          tf_seed = 6,
                          split_fun = function(x) list(x, nn_big_top)#,
                          # df = list(list(c(6,6), 8, 6, 6, 8), 
                          #           list(c(6,6)))
                          )

# freeze_weights(mod_pic$model, from = 1, to = 176)

# k_set_value(mod_int$model$optimizer$lr, 0.01)

mod_pic %>% fit(epochs = 400, 
                view_metrics = FALSE,
                early_stopping = TRUE, 
                patience = 30,
                batch_size = 32,
                validation_split = 0.1)



save_model_weights_hdf5(mod_pic$model, filepath="weights_airbnb_model_pic.hdf5")

# unfreeze_weights(mod_pic$model)
# 
# mod_pic %>% fit(epochs = 100, 
#                 view_metrics = FALSE,
#                 early_stopping = TRUE, 
#                 patience = 10,
#                 batch_size = 32,
#                 validation_split = 0.1)
# 
# save_model_weights_hdf5(mod_pic$model, filepath="weights_airbnb_model_pic_unfreeze.hdf5")

fitted_vals_pic <- mod_pic %>% fitted() %>% unlist()
cor(log(fitted_vals_pic), log(d_train$price))
par(mfrow=c(1,1))
plot(log(fitted_vals_pic), log(d_train$price))
abline(0,1,col="red")
pred_vals_pic <- mod_pic %>% predict(d_test) %>% unlist()

res_df <- rbind(res_df,
                data.frame(model = "images", 
                           cor_train = cor(log(d_train$price), log(fitted_vals_pic)),
                           cor_test = cor(log(d_test$price), log(pred_vals_pic)))
)



res_df %>% xtable()

