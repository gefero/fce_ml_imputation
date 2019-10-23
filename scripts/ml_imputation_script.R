#' ---
#' title: "Introducción (simple) a Machine Learning"
#' subtitle: "Imputando datos perdidos en la Encuesta Permanente de Hogares"
#' author: "Germán Rosati"
#' output: html_notebook
#' ---
#' 
#' ## Objetivos
#' 
#' - Introducir algunos conceptos básicos del enfoque del Aprendizaje Automático
#' - Mostrar el framework `caret` para automatizar algunas tareas del entrenamiento
#' - Aplicar técnicas de ensable para la imputación de datos perdidos en EPH
#' 
#' 
#' ## El problema
#' 
#' Nuestro problema central es, entonces, poder realizar una imputación de los datos perdidos en la variable correspondiente a los ingresos de la ocupación principal (`p21`) en la EPH del segundo trimestre del 2015.
#' 
#' Puede notarse que se trata de un problema bastante amigable, por así decirlo, al enfoque de Machine Learning:
#' 
#' - tenemos un conjunto de casos en los que desconocemos nuestra variable $Y$ 
#' - queremos predecirla
#' - el objetivo principal no es, necesariamente, la interpretabilidad del modelo sino más bien su capacidad predictiva
#' 
#' 
#' ## Preprocesando los datos
#' 
#' Lo primero que tenemos que hacer es importar las librerías con las que vamos a trabajar:
#' 
#' 
## ------------------------------------------------------------------------
library(caret)
library(tidyverse)

#' 
#' 
#' Luego, formateamos un poco algunas etiquetas:
#' 
#' 
## ------------------------------------------------------------------------
load('./data/EPH_2015_II.RData')
data$pp03i<-factor(data$pp03i, labels=c('1-SI', '2-No', '9-NS'))



data$intensi<-factor(data$intensi, labels=c('1-Sub_dem', '2-SO_no_dem', 
                                            '3-Ocup.pleno', '4-Sobreoc',
                                            '5-No trabajo', '9-NS'))

data$pp07a<-factor(data$pp07a, labels=c('0-NC',
                                        '1-Menos de un mes',
                                        '2-1 a 3 meses',
                                        '3-3 a 6 meses',
                                        '4-6 a 12 meses',
                                        '5-12 a 60 meses',
                                        '6-Más de 60 meses',
                                        '9-NS'))

#' 
#' 
#' Y separamos los datos perdidos (nuestros datos desconocidos) de nuestro dataset: 
#' 
#' 
## ------------------------------------------------------------------------
df_imp <- data %>%
        filter(imp_inglab1==1) %>%
        select(-imp_inglab1)

df_train <- data %>%
        filter(imp_inglab1==0) %>%
        select(-imp_inglab1) %>%
        mutate(p21 = case_when(
                        p21==0 ~ 100,
                        TRUE ~ p21))


#' 
#' 
#' #### Algunas cosas a notar
#' 
#' Por un lado, vemos que encadenamos unas cuántas operaciones mediante un operador (`%>%`) llamado `pipe`. El pipe es un símbolo que relaciona dos entidades. Dicho en forma más simple, el pipe de R está en familia con otros operadores más convencionales, como +, - o /. Y al igual que los otros operadores, entrega un resultado en base a los operandos que recibe. 
#' 
#' Ahora bien… ¿Para qué sirve? En resumidas cuentas, hace que el código necesario para realizar una serie de operaciones de transformación de datos sea mucho más simple de escribir y de interpretar.
#' 
#' Repasemos la primer secuencia
#' 
#' - filtramos los datos con algún perdido (`%>% filter(imp_inglab==1)`)
#' - eliminamos la columna identificadora de los casos perdidos (`select(-imp_inglab)`)
#' 
#' 
#' ### Estimando el error de generalización
#' 
#' Recordemos: tenemos muchas formas de estimar el error de generalización (train-test split, cross validation, bootstrap). Usaremos una estrategia de validación cruzada. Vamos a generar los índices mediante `caret`. 
#' 
#' Vamos a tener que fijar dos estrategias de estimación del error: la primera para estimar los hiperparámetros de los modelos y la segunda para la estimación final del error de generalización. En ambos casos, utilizaremos validación cruzada, pero sobre dos muestras diferentes.
#' 
#' Primero, fijamos la semilla aleatoria (para asegurarnos la posibilidad de replicabilidad)
#' 
#' 
## ------------------------------------------------------------------------
set.seed(9183)

#' 
#' 
#' Podemos usar la función `createFolds()` para generar los índices. Aquí, pas
#' 
#' 
## ------------------------------------------------------------------------
cv_index <- createFolds(y = df_train$p21,
                        k=5,
                        list=TRUE,
                        returnTrain=TRUE)

#' 
#' 
#' Aquí usamos tres argumentos:
#' 
#' - `y = df_train$p21`, es el vector de resultados. En nuestro caso, los ingresos de la ocupación principal
#' - `k=5`, es la cantidad de grupos para realizar la validación cruzada
#' - `returnTrain=TRUE`, le decimos que lo que nos devuelva, sean las posiciones de correspondientes a los datos de entrenamiento en cada posición
#' 
#' Finalmente, especificamos el diseño de remuestreo mediante la función `trainControl`:
#' 
#' 
## ------------------------------------------------------------------------
fitControl <- trainControl(
        index=cv_index,
        method="cv",
        number=5,
        verbose = TRUE,
        allowParallel=FALSE)

#' 
#' 
#' En este caso, especificamos los siguientes argumentos:
#' 
#' - `index=cv_index`: el índice que define cuáles casos son de entrenamiento y cuáles de test
#' - `method="cv"`: el método (validación cruzada)
#' 
#' Y generamos los esquemas para la evaluación final:
#' 
#' 
## ------------------------------------------------------------------------
set.seed(7412)
cv_index_final <- createFolds(y = df_train$p21,
                        k=5,
                        list=TRUE,
                        returnTrain=TRUE)

fitControl_final <- trainControl(
        indexOut=cv_index_final, 
        method="cv",
        number=5,
        allowParallel=FALSE)

#' 
#' 
#' ## Entrenando modelos (`train()`)
#' 
#' Tenemos listo nuestro esquema de remuestreo. Podemos pasar a entrenar nuestro primer modelo. Para ello haremos uso extensivo de la función `train()`. La misma puede usarse para 
#' 
#' - evaluar mediante remuestreo el efecto de cada hiperparámetro en la performance
#' - elegir el modelo "óptimo" (la mejor combinación de parámetros) 
#' - estimar la performance del modelo
#' 
#' Primero, debemos elegir el modelo para entrenar. Actualmente, `caret` dispone de 238 modelos disponibles. Puede consultarse [la seccion correspondiente ](http://topepo.github.io/caret/available-models.html) del sitio para mayores detalles. También, llegado el caso, podrían usarse modelos ad-hoc definidos por el usuario.
#' 
#' Comencemos con un modelo simple, pero efectivo: `Random Forest`. Como podrán ver en el sitio, cada modelo puede ser estimado por diferentes implementaciones en diferentes paquetes. Nosotros usaremos la implementación del paquete `ranger` por simplicidad.
#' 
#' 
#' ### Grid search
#' 
#' Si corremos el siguiente código, `caret` va a efectuar el tuneo a partir de una evaluación de una grilla de parámetros predeterminada. 
#' 
#' 
## ----eval=FALSE, include=FALSE-------------------------------------------
## rf_tunning <- train(df$p21 ~ ., data = df_train,
##                  method = "ranger",
##                  trControl = fitControl,
##                  verbose = FALSE)

#' 
#' 
#' Pero también es posible definir una grilla de parámetros:
#' 
#' 
## ------------------------------------------------------------------------
grid <- expand.grid(mtry=c(10, 23, 25),
                    splitrule='variance',
                    min.node.size=c(5, 10, 20))

#' 
#' 
#' Y volvemos a utilizar la función `train`:
#' 
#' 
## ------------------------------------------------------------------------
#t0<-proc.time()
#rf_tunning <- train(p21 ~ ., data = df_train, 
#                 method = "ranger", 
#                 trControl = fitControl,
#                 tuneGrid = grid,
#                 verbose = FALSE)
#proc.time() - t0

rf_tunning <- readRDS('./models/rf_tunning.rds')

#' 
#' 
#' En este caso, hemos realizado una búsqueda exhaustiva, es decir, hemos recorrido la totalidad de la grilla de hiperparámetros y hemos seleccionado el mejor modelo. Como puede verse, esto ha llevado un tiempo de cómputo nada despreciable. 
#' 
#' Es por ello que existe una segunda opción...
#' 
#' 
#' ### Random search
#' 
#' En este caso, en lugar de realizar una búsqueda exhaustiva, podemos reducir notablemente el tiempo de cómputo buscando en una muestra aleatoria de la grilla de hiperparámetros. Para esto, solamente debemos agregar un parámetro en el objeto `fitControl`:
#' 
#' 
## ----warning=TRUE--------------------------------------------------------
#fitControl <- trainControl(
#        index=cv_index, 
#        method="cv",
#        number=5,
#        search = 'random',
#        verbose = TRUE,
#        allowParallel=TRUE)

#' 
#' 
#' Y volvemos a entrenar el modelo:
#' 
#' 
## ------------------------------------------------------------------------
#t0<-proc.time()
#rf_tunning_rand <- train(p21 ~ ., data = df_train, 
#                 method = "ranger", 
#                 trControl = fitControl,
#                 tuneGrid = grid,
#                 tuneLength = 15,
#                 verbose = FALSE)
#proc.time() - t0

#' 
#' 
#' ## Seleccionando el mejor modelo
#' 
#' Una vez finalizado el proceso de tunning de los hiperparámetros, podemos proceder a elegir cuál es el mejor modelo. 
#' 
## ------------------------------------------------------------------------

rf_tunning

#' 
#' 
#' Persistamos el modelo en disco:
#' 
#' 
## ------------------------------------------------------------------------
#saveRDS(rf_tunning, '../models/rf_tunning.rds')

#' 
#' 
#' Podemos realizar un plot del efecto de los hiperparámetros:
#' 
#' 
## ------------------------------------------------------------------------
ggplot(rf_tunning)

#' 
#' 
#' Existen diferentes métricas de selección, las cuales deben ser definidas en la función `train`, usando el argumento `selectionFunction` que puede tomar tres valores:
#' 
#' - `"best"`: se selecciona el mejor modelo con la menor métrica de error (la que usaremos aquí)
#' - `"oneSE"`: utiliza la regla de "un desvío estándar" de [Breiman et al (1986)](https://books.google.com.ar/books/about/Classification_and_Regression_Trees.html?id=JwQx-WOmSyQC&redir_esc=y&hl=es)
#' - `"tolerance`; que busca seleccionar el modelo menos complejo dentro de un margen de tolerancia respecto al mejor modelo
#' 
#' También podrían definirse métodos ad-hoc para esta selección.
#' 
#' 
## ------------------------------------------------------------------------
rf_tunning$bestTune

#' 
#' 
#' ¿Cuál es el mejor modelo (en términos absolutos)?
#' 
#' 
#' ## Realizando la evaluación final
#' 
#' Una vez que hemos seleccionado el mejor modelo, podemos pasar a la evaluación final y persistimos el modelo para reutilizarlo en otras aplicaciones:
#' 
#' 
## ------------------------------------------------------------------------
#rf_final<-train(p21 ~ ., data = df_train,
#                method = "ranger", 
#                trControl = fitControl_final, 
#                verbose = FALSE, 
#                tuneGrid = rf_tunning$bestTune,
#                metric='RMSE')

#saveRDS(rf_final, '../models/rf_final.RDS')

rf_final <- readRDS('./models/rf_final.RDS')

rf_final

#' 
#' 
#' Vemos entonces que el modelo seleccionado performa con un $R^2=0.79$ y un $RMSE=2754$. Solamente nos queda entrenar el modelos sobre la totalidad del dataset de entrenamiento:
#' 
#' 
## ------------------------------------------------------------------------
#rf_final_f<-train(p21~., data=df_train,
#                  method = "ranger",
#                  tuneGrid = rf_tunning$bestTune)

#saveRDS(rf_final_f, './models/rf_final_ff.rds')

rf_final_f <- readRDS('./models/rf_final_ff.rds')

rf_final_f

#' 
#' 
#' ## Obteniendo las predicciones finales
#' 
#' El último paso es obtener las predicciones finales (es decir, nuestras imputaciones). Para ello, llamamos a la función `predict()` que toma como primer argumento al objeto que contiene al modelo final y como segundo argumento el data.frame con los datos a imputar:
#' 
#' 
## ------------------------------------------------------------------------
y_preds_rf <- predict(rf_final, df_imp)


#' 
#' 
#' Comparemos, ahora, las distribuciones de datos imputados por el INDEC (mediante el método Hot Deck) y los que hemos imputado con `ranger`. Para ello, organizamos todo en un data frame que, luego, llevamos al formato tidy.
#' 
#' 
## ------------------------------------------------------------------------
preds <- cbind(y_preds_rf,
               df_imp$p21
)

colnames(preds) <- c('RandomForest', 'Hot_Deck')

preds <- preds %>% as.data.frame() %>% gather(model, value)


#' 
#' 
#' Finalmente, ploteamos un gráfico de densidad para comparar las distribuciones de los casos imputados con ambos métodos.
#' 
#' 
## ------------------------------------------------------------------------
ggplot(preds) +
        geom_density(aes(x=value, fill=model), alpha=0.5)

#' 
#' 
#' # Entrenando otro modelo
#' 
#' La idea ahora es que ustedes entrenen otro modelo. Vamos a entrenar y evaluar `xgBoost`. Para ello, deberán utilizar el método `xgbTree` en la función `train`. Como pueden ver en la documentación, `xgbTree` tiene muchos parámetros para tunear. En este caso, y por cuestiones de tiempo, solamente tunearán dos: `max_depth` y `eta`:
#' 
#' 
## ------------------------------------------------------------------------
xgbGrid <- expand.grid(nrounds = 150,                       
                       max_depth = c(1, 5, 10, 25),
                       colsample_bytree = 1,
                       eta = c(0.3, 0.1, 0.05, 0.01),
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1)

#t0<-proc.time()
#xgb_model<-train(p21 ~ ., data = df_train, 
#                 method = "xgbTree", 
#                 trControl = fitControl, 
#                 verbose = FALSE, 
#                 tuneGrid = xgbGrid,
#                 metric='RMSE')
#proc.time() - t0

#' 
#' 
#' Dado que el modelo puede llevar mucho tiempo de entrenamiento, ya lo hemos preentrenado:
#' 
#' 
## ------------------------------------------------------------------------
xgb_tunning <- readRDS('./models/xgb_tunning.rds')

#' 
#' 
#' De esta forma, podemos repetir todo el proceso:
#' 
#' - Evaluamos el error del modelo seleccionado
#' 
#' 
## ------------------------------------------------------------------------
#xgb_final<-train(p21 ~ ., data = df_train,
#                method = "xgbTree", 
#                trControl = fitControl_final, 
#                verbose = FALSE, 
#                tuneGrid = xgb_model$bestTune,
#                metric='RMSE')

xgb_final <- readRDS('./models/xgb_final.RDS')
#saveRDS(xgb_final, '../models/xgb_final.RDS')

xgb_final

#' 
#' 
#' - Entrenamos el modelo final
#' 
#' 
## ------------------------------------------------------------------------
#xgb_final_f<-train(p21~., data=df_train,
#                  method = "xgbTree",
#                  tuneGrid = xgb_model$bestTune)

#saveRDS(xgb_final_f, '../models/xgb_final_ff.rds')

xgb_final_f <- readRDS('./models/xgb_final_ff.rds')
xgb_final_f

#' 
#' 
#' - Generamos las predicciones finales
#' 
#' 
## ------------------------------------------------------------------------
y_preds_xgb <- predict(xgb_final, df_imp)

#' 
#' 
#' ## Comparando modelos
#' 
#' ¿Cuál de los dos modelos consideran que funciona mejor? ¿Por qué?
#' 
#' Comparemos las distribuciones generadas:
#' 
#' 
## ------------------------------------------------------------------------
preds <- cbind(y_preds_rf,
               y_preds_xgb,
               df_imp$p21
)

colnames(preds) <- c('RandomForest', 'XGBoost', 'Hot_Deck')

preds <- preds %>% as.data.frame() %>% gather(model, value)

ggplot(preds) + 
        geom_density(aes(x=value, fill=model), alpha=0.5)

#' 
